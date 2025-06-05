from runner import Runner
import numpy as np
from utils import load_data

rate = []
total, valid = 0, 0

def frame_select(runner, **data):
    detect_res = detect_data[data["vid"]]["detect"]
    num_limit = num_data[data["qid"]]["maxmin"]
    relevant_idx = set()
    relevant = np.zeros((len(detect_res), len(num_limit)))
    for i, detect in enumerate(detect_res):
        for j, (obj_name, num) in enumerate(num_limit.items()):
            if obj_name not in detect:
                continue
            if "max" in num and detect[obj_name] > num["max"]:
                continue
            if "min" in num and detect[obj_name] < num["min"]:
                continue
            relevant[i][j] = 1
    
    filter_relevant = []
    # 一个条件始终不满足，则删除
    for j in range(relevant.shape[1]):
        if np.sum(relevant[:, j]) == 0:
            continue
        filter_relevant.append(relevant[:, j])
    if len(filter_relevant) != 0:
        filter_relevant = np.stack(filter_relevant, axis=-1)
        filter_relevant = np.all(filter_relevant, axis=-1)
        for i, r in enumerate(filter_relevant):    
            if r:
                for before in range(i - 10, i + 10):
                    if before >= 0 and before < len(detect_res):
                        relevant_idx.add(before)
    if len(relevant_idx) == 0:
        for i in range(0, len(detect_res), 2):
            relevant_idx.add(i)
    else:
        global valid
        valid += 1
        rate.append(len(relevant_idx) / len(detect_res))
        
    global total
    total += 1
    
    return {
        "qid": data["qid"],
        "relevant_idx": list(relevant_idx),
        "last": len(detect_res),
    }


if __name__ == "__main__":
    # exp_name = "0413"
    exp_name = "0522"
    
    dataset_name = "egoschema_subset"
    # dataset_name = "nextmc_test"
    
    output_path = f"./outputs/{exp_name}/select.jsonl"
    # detect_data = load_data("./outputs/0404/yolo.jsonl")
    detect_data = load_data("./outputs/0522/yolo.jsonl")
    # num_data = load_data("./outputs/0411/yolo_maxmin.jsonl")
    num_data = load_data("./outputs/0522/yolo_maxmin_gpt-4o.jsonl")
    runner = Runner(frame_select, output_path, iter_key="qid", dataset=dataset_name)
    runner()
    print(np.mean(rate))
    print(f"valid: {valid/total}({valid}/{total})")
