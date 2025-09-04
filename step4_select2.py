from runner import AsyncRunner
import json
import asyncio
from utils import load_data
from utils import create_model, get_frame_by_idx, make_grid, annote_frame_idx, get_video_size
from pathlib import Path
import numpy as np
from utils import crop_img, parse_json, parse_list
from utils import save_data, print_cfg

example = {
    "frame": [1, "2-4", 5],
    "explain": "put your explaination here",
}

PROMPT = f"""Here is a question related to the video and the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The upper left corner of each frame indicates the current frame number and the total number of frames is [frame_num].
**When selecting frames**:
1. If a frame is irrelevant to the question, omit it.
2. If there are multiple frames with highly similar content that do not introduce new information relevant to answering the question, keep only one such frame.
Finally, output a JSON string, where the "explain" field records your reasoning and analysis process, including the reasons for deleting or retaining frames, and the "frame" field contains a JSON list. Each item in the list represents the frame numbers of segments related to the question. You may use a single number for an individual frame, or a string in the format "start-end" to represent a range of consecutive frames; in the actual output, replace "start" and "end" with the actual start and end frame numbers. Below is the question: [question]. Output example:{json.dumps(example)}, Please provide your answer:
"""


async def frame_select(runner, **data):
    qid = data["qid"]
    video_path = runner.dataset.config.video_path
    video_path = Path(video_path).joinpath(data["video_path"])
    video_size = get_video_size(video_path, 1)
    boxes = []
    last = video_size
    if qid not in results_data or data["qid"] not in select_data:
        print(f"Warning: {qid} not in results or select_data, using {max_frame} frames")
        valid = np.linspace(0, video_size - 1, max_frame).astype(int).tolist()
        valid = sorted(set(valid))
        results = {}    
    else:
        results = results_data[qid]["results"]  
        valid = select_data[data["qid"]]["relevant_idx"]
        valid = [v for v in valid if v >= 0 and v < video_size]
        if len(valid) > max_frame:
            valid_idx = np.linspace(0, len(valid) - 1, max_frame).astype(int).tolist()
            valid = [valid[i] for i in valid_idx]
        # last = results_data[data["qid"]]["last"]
    if uniform_sample:
        valid = np.linspace(0, last - 1, len(valid)).astype(int).tolist()
        valid = sorted(set(valid))
    for i in valid:
        if str(i) not in results:
            boxes.append([])
            continue
        if single_obj:
            boxes.append([b for v in results[str(i)].values() for v2 in v for b in v2["boxes"]])
        else:
            boxes.append(results[str(i)]["boxes"])

    frames = get_frame_by_idx(video_path, valid)
    if use_crop and add_frame_idx:
        images = [
            annote_frame_idx(crop_img(frames[v], boxes[i]), v)
            for i, v in enumerate(valid)
        ]
    elif add_frame_idx:
        images = [annote_frame_idx(frame, v) for frame, v in zip(frames, valid)]
    elif use_crop:
        images = [crop_img(frame, box) for frame, box in zip(frames, boxes)]
    else:
        images = [frame for frame in frames]
    image = make_grid(images, max_frame=max_frame)
    relevant_idx = []
    try:
        question = data["question"]
        if question_only:
            question = data["question"].split("A. ")[0]
        prompt = PROMPT.replace("[question]", question)
        prompt = prompt.replace("[frame_num]", str(last))
        prompt = f"NOTE: you should think step by step before give the final answer.\n{prompt}"
        if save_img:
            save_dir = Path(output_path.replace(".jsonl", "_image"))
            save_dir.mkdir(parents=True, exist_ok=True)
            image.save(str(save_dir.joinpath(f"{qid}.jpg")))
        
        out = await model.forward(prompt, image)
        parsed = parse_json(out)
        relevant_idx = parse_list(parsed["frame"])
        relevant_idx = [idx for idx in relevant_idx if idx < last and idx >= 0]
    except Exception as e:
        print(f"Error processing qid {qid}: {e}")
        return None
    invalid = False
    if len(relevant_idx) == 0:
        relevant_idx = sorted(set(np.linspace(0, last - 1, 24).astype(int).tolist()))
        invalid = True

    return {
        "qid": data["qid"],
        "relevant_idx": relevant_idx,
        **parsed,
        "prompt": prompt,
        "invalid": invalid,
        "input_idx": valid
    }


if __name__ == "__main__":
    cfg = load_data("./configs/select2.yml")
    select_cfg = load_data(cfg["select"])
    dino_cfg = load_data(select_cfg["dino"])
    obj_cfg = load_data(dino_cfg["obj"])

    dataset_name = obj_cfg["dataset_name"]
    single_obj = dino_cfg["single_obj"]  # 是否只使用单个对象
    model_name = cfg["model_name"]
    max_frame = cfg["max_frame"]
    use_crop = cfg.get("use_crop", True)
    uniform_sample = cfg.get("uniform_sample", False)
    add_frame_idx = cfg.get("add_frame_idx", True)
    question_only = cfg.get("question_only", False)
    save_img = cfg.get("save_img", False)
    results_data = load_data(f"./outputs/{dino_cfg['exp_name']}/dino.jsonl")

    exp_name = cfg.get("exp_name")
    if exp_name is None:
        exp_name = select_cfg.get("exp_name")
    if exp_name is None:
        exp_name = dino_cfg.get("exp_name")
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
    cfg["exp_name"] = exp_name

    output_path = f"./outputs/{exp_name}/select2.jsonl"
    save_data(cfg, f"./outputs/{exp_name}/select2.yml")
    print_cfg(cfg)
    select_data = load_data(f"./outputs/{select_cfg['exp_name']}/select.jsonl")

    model = create_model("api", model_name)

    runner = AsyncRunner(
        frame_select,
        output_path,
        iter_key="qid",
        dataset=dataset_name,
    )
    asyncio.run(runner())

    compress_rate = []
    valid_rate = []
    frame_num = []

    out = load_data(output_path)
    for item in runner.dataset:
        if item["qid"] not in out:
            print(f"Warning: {item['qid']} not in output data")
            continue
        if item["qid"] not in select_data:
            print(f"Warning: {item['qid']} not in select data")
            continue
        out_item = out[item["qid"]]
        if out_item["invalid"]:
            valid_rate.append(0)
        else:
            valid_rate.append(1)
            rate = len(out_item["relevant_idx"]) / len(
                select_data[item["qid"]]["relevant_idx"]
            )
            compress_rate.append(rate)
            frame_num.append(len(out_item["relevant_idx"]))

    print(f"compress rate: {np.mean(compress_rate)*100:.2f}%")
    print(f"avg frames: {np.mean(frame_num):.2f}")
    print(f"valid: {np.mean(valid_rate):.2f}({sum(valid_rate)}/{len(valid_rate)})")
