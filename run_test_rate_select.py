from runner import Runner
import numpy as np
from utils import load_data

rate = []
total, valid = 0, 0

def frame_select(**data):
    global total
    total += 1
    if data["qid"] not in select_data2:
        return True
    global valid
    valid += 1
    rate.append(len(select_data2[data["qid"]]["answer"]) / select_data[data["qid"]]["last"])
    return True


if __name__ == "__main__":
    output_path = "./outputs/0415/select.jsonl"
    select_data2 = load_data("./outputs/0413/select2.jsonl")
    select_data = load_data("./outputs/0413/select.jsonl")
    runner = Runner(frame_select, output_path, iter_key="qid")
    runner()
    print(np.mean(rate))
    print(f"valid: {valid/total}({valid}/{total})")
