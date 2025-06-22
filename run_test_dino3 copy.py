from runner import Runner
import numpy as np
from utils import load_data
import torch

rate, frame_num = [], []
total, valid, quest_no_obj, frame_no_obj = 0, 0, 0, 0

def make_exist_table(pred_obj, results):
    frame_obj_map = np.zeros((len(results), len(pred_obj)), dtype=bool)
    for i, result in results.items():
        i = int(i)
        for label in result["labels"]:
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
        while end + 1 < len(frame_obj_map) and np.all(frame_obj_map[end + 1] == frame_obj_map[start]):
            end += 1
        key = start if end == start else f"{start}-{end}"
        exist_table[key] = [pred_obj[i] for i, exist in enumerate(frame_obj_map[start]) if exist]
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

def frame_select(runner, **data):
    pred_obj = detect_data[data["qid"]]["pred"]["question"]
    results = input_data[data["qid"]]["results"]
    frame_obj_map = np.zeros((len(results), len(pred_obj)), dtype=bool)
    relevant_idx = []
    
    for i, result in results.items():
        i = int(i)
        if len(result["labels"]) == 0:
            
            continue # no objects detected, skip this image
        if not skip_object:    
            if apear_any or (len(result["labels"]) == len(pred_obj)):
                relevant_idx.append(i)
            continue
        # record objects that appear in the image
        for label in result["labels"]:
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
    
    if skip_object:
        # skip objects that not apear in all images
        legal_obj = np.any(frame_obj_map, axis=0)
        frame_obj_map = frame_obj_map[:, legal_obj]
        if not apear_any:
            relevant_idx = np.all(frame_obj_map, axis=1)
        else:
            relevant_idx = np.any(frame_obj_map, axis=1)
        relevant_idx = np.nonzero(relevant_idx)[0].tolist()
               
    if len(relevant_idx) == 0:
        if len(pred_obj) == 0:
            global quest_no_obj
            quest_no_obj += 1
        else:
            global frame_no_obj
            frame_no_obj += 1
        relevant_idx = sorted(list(set(np.linspace(0, len(results)-1, 8).astype(int).tolist())))
    else:
        global valid
        valid += 1
        rate.append(len(relevant_idx) / len(results))
        frame_num.append(len(relevant_idx))
        
    global total
    total += 1
    
    return {
        "qid": data["qid"],
        "relevant_idx": relevant_idx,
        "last": len(results),
    }


if __name__ == "__main__":
    # exp_name = "0601"
    exp_name = "0607"
    
    skip_object = True # skip objects that not apear in all images
    # skip_object = False
    
    # apear_any = True # select frames that apear any object
    apear_any = False
    
    # dataset_name = "egoschema_subset"
    dataset_name = "nextmc_test"
    
    detect_data = load_data("./outputs/0604/dino_gpt-4.1_nextmc_test_option2.jsonl")
    # detect_data = load_data("./outputs/0607/dino_gpt-4.1_egoschema_subset_option2.jsonl")
    input_data = load_data("./outputs/0607/dino_out_nextmc_test.jsonl")
    # input_data = load_data("./outputs/0607/dino_out_egoschema_subset.jsonl")
    
    end = "_skip" if skip_object else ""
    end += "_any" if apear_any else ""
    output_path = f"./outputs/{exp_name}/dino_select_{dataset_name}{end}.jsonl"
    
    runner = Runner(frame_select, output_path, iter_key="qid", dataset=dataset_name)
    runner()
    print(f"compress rate: {np.mean(rate)}")
    print(f"avg frames: {np.mean(frame_num)}")
    print(f"valid: {valid/total}({valid}/{total})")
    print(f"quest_no_obj: {quest_no_obj/total}({quest_no_obj}/{total})")
    print(f"frame_no_obj: {frame_no_obj/total}({frame_no_obj}/{total})")
