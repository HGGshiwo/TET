from runner import AsyncRunner
import json
import asyncio
from utils import load_data
from utils import (
    parse_json,
    create_model,
    get_frame,
    make_grid,
    annote_frame_idx,
)
from pathlib import Path
import numpy as np
from utils import crop_img, make_anno_grid
from utils import save_data

example = {
    "answer": "A",
    "explain": "put your explaination here",
    "confidence": 3,
}

PROMPT1 = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT1_anno = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The important areas related to the question in the frame are enclosed by red boxes. You should pay special attention to these parts. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT1_cont = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Each frame is also shown with an additional image on its right side, which is a zoomed-in and cropped version highlighting a key object extracted from that frame. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT1_frame_idx = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The upper left corner of each frame indicates the current frame number and the total number of frames is [frame_num]. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"


async def frame_select(runner, **data):
    qid = data["qid"]
    if data["qid"] not in select_data2:
        last = results_data[data["qid"]]["last"]
        valid = np.linspace(0, last - 1, max_frame).astype(int).tolist()
        valid = list(set(valid))
    else:
        valid = select_data2[data["qid"]]["relevant_idx"]
        valid = [v for v in valid if v < results_data[data["qid"]]["last"] and v >= 0]

    if uniform_sample or add_frame_idx:
        last = results_data[data["qid"]]["last"]
    if uniform_sample:
        frame_num = len(valid)
        if uniform_target == "both":    
            valid = np.linspace(0, last - 1, frame_num).astype(int).tolist()
            valid = list(set(valid))
        elif uniform_target == "step1":
            pass # valid 需要在select2的输入进行采样，这里仍然使用select2输出的
        elif uniform_target == "step2":
            select1_valid = select_data1[data["qid"]]["relevant_idx"]
            valid_idx = np.linspace(0, len(select1_valid) - 1, frame_num).astype(int).tolist()
            valid_idx = list(set(valid_idx))
            valid = [select1_valid[i] for i in valid_idx]
        else:
            raise ValueError("uniform_target must be one of 'both', 'step1', 'step2'")

    image = None
    video_path = runner.dataset.config.video_path
    video_path = Path(video_path).joinpath(data["video_path"])
    frames = get_frame(video_path, 1)
    if (use_crop or use_anno or use_cont) and (results := results_data[qid]["results"]):
        boxes = []
        for i in valid:
            if str(i) not in results:
                boxes.append([])
                continue
            if single_obj:
                boxes.append([b for v in results[str(i)].values() for b in v["boxes"]])
            else:
                boxes.append(results[str(i)]["boxes"])
        if use_crop:
            image = make_grid(
                [crop_img(frames[v], boxes[i]) for i, v in enumerate(valid)],
                max_frame=max_frame,
            )
        elif use_anno:
            image = make_anno_grid(
                [frames[i] for i in valid], boxes, max_frame=max_frame
            )
        elif use_cont:
            image = []
            for i, (v, b) in enumerate(zip(valid, boxes)):
                if i % 2 == 0:
                    image.append(frames[v])
                else:
                    image.append(crop_img(frames[v], b))
            image = make_grid(image, max_frame=max_frame)
    elif add_frame_idx:
        image = make_grid([annote_frame_idx(frames[i], i) for i in valid])
    else:
        frames = [frames[i] for i in valid]
        image = make_grid(frames, max_frame)
    save_dir = Path(output_path.replace(".jsonl", "_image"))
    save_dir.mkdir(parents=True, exist_ok=True)
    image.save(str(save_dir.joinpath(f"{qid}.jpg")))
    if use_anno:
        prompt = PROMPT1_anno
    elif use_cont:
        prompt = PROMPT1_cont
    elif add_frame_idx:
        prompt = PROMPT1_frame_idx.replace("[frame_num]", str(last))
    else:
        prompt = PROMPT1
    question = data["question"]
    prompt = prompt.replace("[question]", question)
    if use_cot:
        prompt = f"NOTE: you should think step by step before give the final answer.\n{prompt}"
    try:
        out = await model.forward(prompt, image)
        assert out is not None, "model output is None"
        out_raw = out
        out = parse_json(out)
        if not out:
            out = None
            print("model output is None")
        else:
            out["qid"] = qid
            out["prompt"] = prompt
            out["truth"] = data["truth"]
            out["raw"] = out_raw
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":

    cfg = load_data("./configs/answer.yml")
    select2_cfg = load_data(cfg["select2"])
    select_cfg = load_data(select2_cfg["select"])
    dino_cfg = load_data(select_cfg["dino"])
    obj_cfg = load_data(dino_cfg["obj"])

    single_obj = dino_cfg["single_obj"]  # 是否只使用单个对象
    dataset_name = obj_cfg["dataset_name"]
    uniform_sample = cfg["uniform_sample"]  # 是否均匀采样
    if uniform_sample:
        uniform_target = cfg["uniform_target"]

    use_crop = cfg["use_crop"]  # 是否裁剪图片
    use_anno = cfg["use_anno"]  # 是否使用标注的图片
    use_cont = cfg["use_cont"]  # 使用拼接
    use_cot = cfg.get("use_cot", False)  # 是否使用链式推理
    add_frame_idx = cfg["add_frame_idx"]

    assert not (
        use_crop and use_anno and use_cont
    ), "use_crop and use_anno and use_cont cannot be both True"

    max_frame = cfg["max_frame"]
    model_name = cfg["model_name"]

    results_data = load_data(f"./outputs/{dino_cfg['exp_name']}/dino.jsonl")

    exp_name = cfg.get("exp_name")
    if exp_name is None:
        exp_name = select2_cfg.get("exp_name")
    if exp_name is None:
        exp_name = select_cfg.get("exp_name")
    if exp_name is None:
        exp_name = dino_cfg.get("exp_name")
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
    cfg["exp_name"] = exp_name

    output_path = f"./outputs/{exp_name}/answer.jsonl"
    save_data(cfg, f"./outputs/{exp_name}/answer.yml")

    select_data2 = load_data(f"./outputs/{select2_cfg['exp_name']}/select2.jsonl")
    select_data1 = load_data(f"./outputs/{select_cfg['exp_name']}/select.jsonl")
    
    model = create_model("api", model_name)

    runner = AsyncRunner(
        frame_select,
        output_path,
        iter_key="qid",
        dataset=dataset_name,
    )
    asyncio.run(runner())

    # compute metrics
    result = load_data(output_path)
    compute_metrics = runner.dataset.get_compute_metrics2()
    total, difficult = 0, 0
    for item in runner.dataset:
        total += 1
        if item["qid"] not in result:
            continue
        if "pred" in result[item["qid"]]:
            out = compute_metrics(result[item["qid"]]["pred"], item, True)
        else:
            out = compute_metrics(result[item["qid"]]["answer"], item, True)
    failed = out.pop("failed")
    failed_path = output_path.replace(".jsonl", ".txt")
    Path(failed_path).write_text("\n".join(failed))
    print(out)
