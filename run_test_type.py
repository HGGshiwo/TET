from utils import load_data
from runner import Runner

q_type = "T"
dataset_name = "nextmc_test"
data_path = "./outputs/exp0623_tem2/answer.jsonl"
data = load_data(data_path)

def frame_select(runner, **item):
    answer = data[item["qid"]]["answer"]
    if item["q_type"][0] == q_type and item["truth"] != answer:
        print(item["qid"])

Runner(
    frame_select,
    None,
    iter_key="qid",
    dataset=dataset_name,
)()