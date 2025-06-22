from runner import AsyncRunner
import numpy as np
from utils import load_data
import torch
from task_utils import parse_json, create_model
import asyncio

PROMPT = """
Below is a question related to a video, followed by an object existence table. Each row of the table represents a video segment, with the numbers at the front indicating the start and end frame numbers of that segment. If the segment contains only one frame, the number represents that single frame. The following strings indicate the list of objects present in that segment. Based on the question and the object existence table, please analyze which segments (frame numbers) are relevant to the question. Finally, output a JSON string, where the "explain" field records your reasoning and analysis process, and the "frame" field contains a JSON list. Each item in the list represents the frame numbers of segments related to the question. You may use a single number for an individual frame, or a string in the format "start-end" to represent a range of consecutive frames; in the actual output, replace "start" and "end" with the actual start and end frame numbers. Below is the question: [question]. Below is the object existence table: [table]. Please provide your answer.
"""


def parse_list(list_data):
    out = set()
    for d in list_data:
        if isinstance(d, str):
            if "-" in d:
                start, end = map(int, d.split("-"))
                out.update(list(range(start, end + 1)))
            else:
                out.add(int(d))
        else:
            assert isinstance(d, int), f"Invalid type in list_data: {d}"
            out.add(d)
    return sorted(list(out))


def make_exist_table(pred_obj, results):
    frame_obj_map = np.zeros((len(results), len(pred_obj)), dtype=bool)
    for i, result in results.items():
        i = int(i)
        # for label in result["labels"]:
        for label in result.keys():
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
        try:
            prompt = PROMPT.replace("[question]", data["question"].split("A. ")[0])
            table = "\n".join(
                [
                    f"{key}: {','.join(value)}"
                    for key, value in exist_table.items()
                    if len(value) > 0
                ]
            )
            prompt = prompt.replace("[table]", table)
            out = await model.forward(prompt)
            parsed = parse_json(out)
            relevant_idx = parse_list(parsed["frame"])
            if len(relevant_idx) == 0:
                invalid_type = "model_no_obj"
        except Exception as e:
            print(f"{data['qid']} error: {e}")
            return None

    if len(relevant_idx) == 0:
        relevant_idx = sorted(
            list(set(np.linspace(0, last - 1, 8).astype(int).tolist()))
        )

    return {
        "qid": data["qid"],
        "relevant_idx": relevant_idx,
        **parsed,
        "prompt": prompt,
        "invalid_type": invalid_type,
    }


if __name__ == "__main__":
    # exp_name = "0601"
    # exp_name = "0607"
    # exp_name = "0614"
    exp_name = "0621"
    
    model_name = "gpt-4.1"

    # dataset_name = "egoschema_subset"
    dataset_name = "nextmc_test"

    detect_data = load_data("./outputs/0604/dino_gpt-4.1_nextmc_test_option2.jsonl")
    # detect_data = load_data("./outputs/0607/dino_gpt-4.1_egoschema_subset_option2.jsonl")
    # input_data = load_data("./outputs/0607/dino_out_nextmc_test.jsonl")
    # input_data = load_data("./outputs/0609/dino_out_egoschema_subset.jsonl")
    # input_data = load_data(f"./outputs/0619/dino_out_nextmc_test_low_base.jsonl")
    input_data = load_data(f"./outputs/0619/dino_out_nextmc_test_low_tiny.jsonl")
    
    output_path = f"./outputs/{exp_name}/dino_select_{model_name}_{dataset_name}.jsonl"

    model = create_model("api", model_name)

    runner = AsyncRunner(
        frame_select, output_path, iter_key="qid", dataset=dataset_name
    )
    asyncio.run(runner())
    
    rate, frame_num = [], []
    total, valid, quest_no_obj, frame_no_obj, model_no_obj = 0, 0, 0, 0, 0

    out_data = load_data(output_path)
    for item in runner.dataset:
        results = input_data[item["qid"]]["results"]
        pred_obj = detect_data[item["qid"]]["pred"]["question"]
        exist_table = make_exist_table(pred_obj, results)
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
