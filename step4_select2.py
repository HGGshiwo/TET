from runner import AsyncRunner
import json
import asyncio
from utils import load_data
from utils import (
    create_model,
    get_frame_by_idx,
    make_grid,
    annote_frame_idx,
    get_video_size,
)
from pathlib import Path
import numpy as np
from utils import parse_json, parse_list, chunk
from utils import save_data, print_cfg

example = {
    "frame": [1, 2, 3, 4, 5],
    "explain": "put your explaination here",
}

PROMPT = f"""Here is a question related to the video and the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The upper left corner of each frame indicates the current frame number and the total number of frames is [frame_num].
**When selecting frames**:
1. If a frame is irrelevant to the question, omit it.
2. If there are multiple frames with highly similar content that do not introduce new information relevant to answering the question, keep only one such frame.
3. The maximum number of frames you are allowed to select is [max_frame].
Finally, output a JSON string, where the "explain" field records your reasoning and analysis process, including the reasons for deleting or retaining frames, and the "frame" field contains a JSON list. Each item in the list represents the frame numbers of segments related to the question. Below is the question: [question]. Output example:{json.dumps(example)}, Please provide your answer:
"""


async def frame_select(runner, **data):
    max_frame = runner.max_frame
    qid = data["qid"]
    video_path = runner.dataset.config.video_path
    video_path = Path(video_path).joinpath(data["video_path"])
    video_size = get_video_size(video_path, 1)
    last = video_size
    if qid not in results_data or data["qid"] not in select_data:
        print(f"Warning: {qid} not in results or select_data, using {max_frame} frames")
        valid = np.linspace(0, video_size - 1, max_frame).astype(int).tolist()
        valid = sorted(set(valid))
    else:
        valid = select_data[data["qid"]]["relevant_idx"]
        valid = [v for v in valid if v >= 0 and v < video_size]
        if len(valid) > max_frame:
            valid_idx = np.linspace(0, len(valid) - 1, max_frame).astype(int).tolist()
            valid = [valid[i] for i in valid_idx]
        # last = results_data[data["qid"]]["last"]
    if uniform_sample:
        valid = np.linspace(0, last - 1, len(valid)).astype(int).tolist()
        valid = sorted(set(valid))

    relevant_idx = []
    question = data["question"]
    if question_only:
        question = data["question"].split("A. ")[0]
    prompt = PROMPT.replace("[question]", question)
    prompt = prompt.replace("[frame_num]", str(last))
    prompt = prompt.replace("[max_frame]", str(max_frame // 2))
    prompt = (
        f"NOTE: you should think step by step before give the final answer.\n{prompt}"
    )
    _parsed = []
    
    if save_img:
        save_dir = Path(output_path.replace(".jsonl", "_image"))
        save_dir.mkdir(parents=True, exist_ok=True)
                    
    for i in range(0, len(valid), frame_per_req):
        chunk_valid = valid[i : i + frame_per_req]
        j = i + len(chunk_valid) - 1
        frames = get_frame_by_idx(video_path, valid)
        if add_frame_idx:
            images = [
                annote_frame_idx(frame, v) for frame, v in zip(frames, chunk_valid)
            ]
        else:
            images = [frame for frame in frames]
        try:
            input_images = []
            for _i, chunk_img in enumerate(chunk(images, frame_per_img)):
                image = make_grid(chunk_img, max_frame=frame_per_img)
                if save_img:
                    image.save(str(save_dir.joinpath(f"{qid}_{valid[i]}-{valid[j]}_chunk{_i}.jpg")))
                input_images.append(image)
            out = await model.forward(prompt, input_images)
            parsed = parse_json(out)
            cur_relevant_idx = parse_list(parsed["frame"])
            cur_relevant_idx = [
                idx for idx in cur_relevant_idx if idx <= valid[j] and idx >= valid[i]
            ]
            relevant_idx.extend(cur_relevant_idx)
            _parsed.append(parsed)
        except Exception as e:
            print(f"Error processing qid {qid}: {e}")
            return None
    invalid = False
    if len(relevant_idx) == 0:
        relevant_idx = sorted(
            set(np.linspace(0, last - 1, max_frame).astype(int).tolist())
        )
        invalid = True

    return {
        "qid": data["qid"],
        "relevant_idx": relevant_idx,
        "raw_output": _parsed,
        "prompt": prompt,
        "invalid": invalid,
        "input_idx": valid,
    }


if __name__ == "__main__":
    cfg = load_data("./configs/select2.yml")
    select_cfg = load_data(cfg["select"])
    dino_cfg = load_data(select_cfg["dino"])
    obj_cfg = load_data(dino_cfg["obj"])

    dataset_name = obj_cfg["dataset_name"]
    model_name = cfg["model_name"]
    max_frame = cfg["max_frame"]
    frame_per_req = cfg["frame_per_req"]
    frame_per_img = cfg["frame_per_img"]
    use_crop = cfg.get("use_crop", True)
    uniform_sample = cfg.get("uniform_sample", False) # 对前一步进行消融
    add_frame_idx = cfg.get("add_frame_idx", True)
    question_only = cfg.get("question_only", False)
    save_img = cfg.get("save_img", False)
    iter_num = cfg["iter_num"]

    results_data = load_data(f"./outputs/{dino_cfg['exp_name']}/dino.jsonl")

    exp_name = cfg.get("exp_name")
    if exp_name is None:
        exp_name = select_cfg.get("exp_name")
    if exp_name is None:
        exp_name = dino_cfg.get("exp_name")
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
    cfg["exp_name"] = exp_name

    save_data(cfg, f"./outputs/{exp_name}/select2.yml")
    print_cfg(cfg)

    model = create_model("api", model_name)

    last_data_path = f"./outputs/{select_cfg['exp_name']}/select.jsonl"
    
    output_path_list = []
    select_data_list = [f"./outputs/{select_cfg['exp_name']}/select.jsonl"]
    for i in range(iter_num):
        output_path = f"./outputs/{exp_name}/select2.jsonl"
        if i != iter_num - 1:
            output_path = output_path.replace(".jsonl", f"_{i+1}.jsonl")
        output_path_list.append(output_path)
        if i != iter_num - 1:
            select_data_list.append(output_path)
            
    for i in range(iter_num):
        print(f"Select frame round: [{i + 1}/{iter_num}]")
        output_path = output_path_list[i]
        select_data = load_data(select_data_list[i])
        runner = AsyncRunner(
            frame_select,
            output_path,
            iter_key="qid",
            dataset=dataset_name,
            max_frame=max_frame,
        )
        uniform_sample = False # 只对第一步进行消融
        asyncio.run(runner())
        max_frame = max_frame // 2
    
    compress_rates = [[] for _ in range(iter_num)]
    valid_rates = [[] for _ in range(iter_num)]
    frame_nums = [[] for _ in range(iter_num)]
    outs = [load_data(output_path) for output_path in output_path_list]
    select_datas = [load_data(select_data_list[0])] + outs[:-1]
    for i in range(iter_num):    
        out = outs[i]
        select_data = select_datas[i]
        compress_rate = compress_rates[i]
        valid_rate = valid_rates[i]
        frame_num = frame_nums[i]
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
    for i in range(iter_num):
        compress_rate = compress_rates[i]
        frame_num = frame_nums[i]
        valid_rate = valid_rates[i]
        print(f"Iter {i+1}:")
        print(f"compress rate: {np.mean(compress_rate)*100:.2f}%")
        print(f"avg frames: {np.mean(frame_num):.2f}")
        print(f"valid: {np.mean(valid_rate):.2f}({sum(valid_rate)}/{len(valid_rate)})")
