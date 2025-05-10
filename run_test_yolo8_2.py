""" 
1. 模型直接根据图片分析答案
2. uniform_sample = True, 表示消融实验
"""
import asyncio
from utils import load_data
from task_utils import api_forward, create_model
import numpy as np
from runner import AsyncRunner, Runner

PROMPT = "Here is a question related to the picture: [question]. Depending on the content of the picture, which option is most likely to be correct, allowing 0 or more possible options to be selected. Give your reasons"


def filter(data):
    last = select_data[data["qid"]]["last"]
    if data["qid"] in select_data2:
        if len(select_data2[data["qid"]]["answer"]) == 0:
            return data["idx"] % 2 == 0
        relevant = select_data2[data["qid"]]["answer"]
        if uniform_sample:
            length = len(relevant)
            relevant = np.linspace(0, last, length + 1).astype(int).tolist()
        return data["idx"] in relevant
    elif data["qid"] in select_data:
        if len(select_data[data["qid"]]["relevant_idx"]) == 0:
            return data["idx"] % 2 == 0
        relevant = select_data[data["qid"]]["relevant_idx"]
        if uniform_sample:
            length = len(select_data[data["qid"]]["relevant_idx"])
            relevant = np.linspace(0, last, length + 1).astype(int).tolist()
        return data["idx"] in relevant
    return False


def frame_select(runner, **data):
    qid = data["qid"]
    prompt = PROMPT.replace("[question]", data["question"])
    try:
        out = model.forward(prompt, data["frame"])
        out = {"answer": out, "qid": qid, "idx": data["idx"], "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out

def frame_select2(runner, batch_data):
    batch_prompt, batch_frame, batch_out = [], [], []
    for data in batch_data:
        qid = data["qid"]
        prompt = PROMPT.replace("[question]", data["question"])
        batch_prompt.append(prompt)
        batch_frame.append(data["frame"])
    try:
        out = model.forward(batch_prompt, batch_frame)
    except Exception as e:
        print(e)
        out = None
    for data, o, prompt in zip(batch_data, out, batch_prompt):
        qid = data["qid"]
        batch_out.append({"answer": o, "qid": qid, "idx": data["idx"], "prompt": prompt})
    return batch_out

async def async_frame_select(runner, **data):
    qid = data["qid"]
    prompt = PROMPT.replace("[question]", data["question"])
    try:
        out = await api_forward(prompt, data["frame"])
        out = {"answer": out, "qid": qid, "idx": data["idx"], "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0502/answer2_2.jsonl"
    select_data = load_data("./outputs/0413/select.jsonl")
    select_data2 = load_data("./outputs/relevant.jsonl")
    batch_size = 8
    use_api = True
    use_qwen = True
    uniform_sample = False
    
    if not use_api:
        model_type = "llava" if not use_qwen else "qwen"
        model = create_model(model_type)

    task_func = None
    if use_api:
        task_func = async_frame_select 
        batch_size = 1
    else:
        if batch_size == 1:
            task_func = frame_select
        else:
            task_func = frame_select2
    runner_cls = AsyncRunner if use_api else Runner
    runner = runner_cls(
        task_func,
        output_path,
        iter_key="qid",
        iter_frame=True,
        video_fps=1,
        filter=filter,
        batch_size=batch_size,
    )

    if not use_api:
        runner()
    else:
        asyncio.run(runner())
