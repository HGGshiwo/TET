from runner import AsyncRunner
import numpy as np
from utils import load_data, load_data, save_data
import torch
from utils import parse_json, create_model, parse_list, print_cfg
import asyncio

PROMPT_v1 = """
Below is a question related to a video, followed by an object existence table. Each row of the table represents a video segment, with the numbers at the front indicating the start and end frame numbers of that segment. If the segment contains only one frame, the number represents that single frame. The following strings indicate the list of objects present in that segment. Based on the question and the object existence table, please analyze which segments (frame numbers) are relevant to the question. Finally, output a JSON string, where the "explain" field records your reasoning and analysis process, and the "frame" field contains a JSON list. Each item in the list represents the frame numbers of segments related to the question. You may use a single number for an individual frame, or a string in the format "start-end" to represent a range of consecutive frames; in the actual output, replace "start" and "end" with the actual start and end frame numbers. Below is the question: [question]. Below is the object existence table: [table]. Please provide your answer.
"""
# 增加对start, middle, end的解释
PROMPT_v2 = """
Below is a question related to a video, followed by an object existence table. Each row of the table represents a video segment, with the numbers at the front indicating the start and end frame numbers of that segment. If the segment contains only one frame, the number represents that single frame. The following strings indicate the list of objects present in that segment. Based on the question and the object existence table, please analyze which segments (frame numbers) are relevant to the question. If the problem includes descriptions like "start," "middle," or "end" referring to the position in the video, it refers to the first 30%, the middle 30%, and the last 30% of the video, respectively. Finally, output a JSON string, where the "explain" field records your reasoning and analysis process, and the "frame" field contains a JSON list. Each item in the list represents the frame numbers of segments related to the question. You may use a single number for an individual frame, or a string in the format "start-end" to represent a range of consecutive frames; in the actual output, replace "start" and "end" with the actual start and end frame numbers. Below is the question: [question]. Below is the object existence table: [table]. Please provide your answer.
"""

PROMPT_v3 = """
Below is a question related to a video, followed by an object existence table. Each row of the table represents a video segment, with the numbers at the front indicating the start and end frame numbers of that segment. If the segment contains only one frame, the number represents that single frame. The following strings indicate the list of objects present in that segment. Based on the question and the object existence table, please analyze which segments (frame numbers) are relevant to the question. Here are the things you need to pay attention to:
1. If the problem includes descriptions like "start," "middle," or "end" referring to the position in the video, it refers to the first 30%, the middle 30%, and the last 30% of the video, respectively. 
2. The existence table is constructed by the object detection model and might not be completely accurate. For example, if the same object appears in both the previous and next frames, but is missing in the middle frame, it could be due to a missed detection.
3. You only need to exclude frames that are clearly irrelevant to the problem, and make sure not to miss any frames that might be related. Don't worry about retaining too many frames, as further filtering will be performed in subsequent steps.
Finally, output a JSON string, where the "explain" field records your reasoning and analysis process, and the "frame" field contains a JSON list. Each item in the list represents the frame numbers of segments related to the question. You may use a single number for an individual frame, or a string in the format "start-end" to represent a range of consecutive frames; in the actual output, replace "start" and "end" with the actual start and end frame numbers. Below is the question: [question]. Below is the object existence table: [table]. Please provide your answer.
"""


def make_exist_table(pred_obj, results):
    if len(results) == 0:
        return {}
    frame_obj_map = np.zeros(
        (max(map(int, results.keys())) + 1, len(pred_obj)), dtype=bool
    )
    for i, result in results.items():
        i = int(i)
        if single_obj:
            labels = result.keys()
        else:
            labels = result["labels"]
        for label in labels:
            try:
                idx = pred_obj.index(label.lower())
            except ValueError:
                idx = -1
                for j, obj in enumerate(pred_obj):
                    if label.lower() in obj:
                        idx = j
                        break
            if idx == -1:
                continue
            frame_obj_map[i, idx] = True
    exist_table = {}
    start, end = 0, 0
    while start < len(frame_obj_map):
        while end + 1 < len(frame_obj_map) and np.all(
            frame_obj_map[end + 1] == frame_obj_map[start]
        ):
            end += 1
        key = start if end == start else f"{start}-{end}"
        exist_table[key] = [
            pred_obj[i] for i, exist in enumerate(frame_obj_map[start]) if exist
        ]
        start = end + 1
        end = start
    return exist_table


def tensor_to_list(result):
    out = {}
    for key in result.keys():
        if isinstance(result[key], torch.Tensor):
            out[key] = result[key].cpu().numpy().tolist()
        else:
            out[key] = result[key]
    return out


async def frame_select(runner, **data):
    try:
        if data["qid"] not in detect_data:
            pred_obj = []
        else:
            pred_obj = detect_data[data["qid"]]["pred"]["question"]
        last = input_data[data["qid"]]["last"]
        results = input_data[data["qid"]]["results"]
        relevant_idx = []
        exist_table = make_exist_table(pred_obj, results)
        parsed = {}
        prompt = ""
        invalid_type = None
        if len(results) == 0:
            invalid_type = "quest_no_obj"
        elif np.all([len(v) == 0 for v in exist_table.values()]):
            invalid_type = "frame_no_obj"
        else:
            prompt = globals()[f"PROMPT_{prompt_version}"]
            prompt = prompt.replace("[question]", data["question"].split("A. ")[0])
            table = "\n".join(
                [
                    f"{key}: {','.join(value)}"
                    for key, value in exist_table.items()
                    if len(value) > 0
                ]
            )
            prompt = prompt.replace("[table]", table)
            prompt = f"NOTE: you should think step by step before give the final answer.\n{prompt}"
            out = await model.forward(prompt)
            parsed = parse_json(out)
            relevant_idx = parse_list(parsed["frame"])
            if len(relevant_idx) == 0:
                invalid_type = "model_no_obj"
    except Exception as e:
        print(f"{data['qid']} error: {e}")
        return None

    if len(relevant_idx) == 0:
        relevant_idx = sorted(set(np.linspace(0, last - 1, 24).astype(int).tolist()))

    return {
        "qid": data["qid"],
        "relevant_idx": relevant_idx,
        **parsed,
        "prompt": prompt,
        "invalid_type": invalid_type,
    }


if __name__ == "__main__":
    cfg = load_data("./configs/select.yml")
    dino_cfg = load_data(cfg["dino"])
    obj_cfg = load_data(dino_cfg["obj"])
    model_name = cfg["model_name"]
    dataset_name = obj_cfg["dataset_name"]
    single_obj = dino_cfg["single_obj"]  # 是否只使用单个对象
    prompt_version = cfg.get("prompt_version", "v1")
    detect_data = load_data(f"./outputs/{obj_cfg['exp_name']}/obj.jsonl")
    input_data = load_data(f"./outputs/{dino_cfg['exp_name']}/dino.jsonl")

    exp_name = cfg.get("exp_name")
    if exp_name is None:
        exp_name = dino_cfg.get("exp_name")
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
    cfg["exp_name"] = exp_name
    save_data(cfg, f"./outputs/{exp_name}/select.yml")
    print_cfg(cfg)
    output_path = f"./outputs/{exp_name}/select.jsonl"

    model = create_model("api", model_name)

    runner = AsyncRunner(
        frame_select, output_path, iter_key="qid", dataset=dataset_name
    )
    asyncio.run(runner())

    rate, frame_num = [], []
    total, valid, quest_no_obj, frame_no_obj, model_no_obj = 0, 0, 0, 0, 0

    out_data = load_data(output_path)
    for item in runner.dataset:
        if item["qid"] not in input_data:
            print(f"Warning: {item['qid']} not in input data")
            continue
        results = input_data[item["qid"]]["results"]
        total += 1
        if item["qid"] not in out_data:
            print(f"Warning: {item['qid']} not in output data")
            continue
        out = out_data[item["qid"]]
        if out["invalid_type"] == "quest_no_obj":
            quest_no_obj += 1
        elif out["invalid_type"] == "frame_no_obj":
            frame_no_obj += 1
        elif out["invalid_type"] == "model_no_obj":
            model_no_obj += 1
        else:
            valid += 1
            rate.append(len(out["relevant_idx"]) / len(results))
            frame_num.append(len(out["relevant_idx"]))
    print(f"compress rate: {np.mean(rate)}")
    print(f"avg frames: {np.mean(frame_num)}")
    print(f"valid: {valid/total}({valid}/{total})")
    print(f"quest_no_obj: {quest_no_obj/total}({quest_no_obj}/{total})")
    print(f"frame_no_obj: {frame_no_obj/total}({frame_no_obj}/{total})")
