from runner import Runner
import numpy as np
from utils import load_data

rate, frame_num = [], []
total, valid = 0, 0

def frame_select(**data):
    global total
    total += 1
    if data["qid"] not in select_data2:
        return True
    global valid
    valid += 1
    if "last" in select_data2[data["qid"]]:
        last = select_data2[data["qid"]]["last"]
    elif "last" in  select_data[data["qid"]]:
        last = select_data[data["qid"]]["last"]
    rate.append(len(select_data2[data["qid"]]["answer"]) / last)
    frame_num.append(len(select_data2[data["qid"]]["answer"]))
    return True


if __name__ == "__main__":
    output_path = "./outputs/0415/select.jsonl"
    # select_data2 = load_data("./outputs/0413/select2.jsonl")
    select_data = load_data("./outputs/0529/clip_clip2_lvnet_limit.jsonl")
    select_data2 = load_data("")
    runner = Runner(frame_select, output_path, iter_key="qid")
    runner()
    print(f"compress rate: {np.mean(rate)}")
    print(f"valid: {valid/total}({valid}/{total})")
