""" 
1. 模型直接根据图片分析答案
2. uniform_sample = True, 表示消融实验
"""
import asyncio
from utils import load_data
from task_utils import create_model
import numpy as np
from runner import AsyncRunner, Runner

PROMPT1 = "Here is a question related to the picture: [question]. Depending on the content of the picture, which option is most likely to be correct, allowing 0 or more possible options to be selected. Give your reasons"

PROMPT2 = "Please provide a brief answer to the following questions: [question], output the answer directly without other text, if there is not enough information in the frame, just answer 'not know', seperate an with line breaks. Output Example:\n1: answer1\n2: answer2\n3: answer3"

def parse(pred, num):
    try:
        answer = [o.split(":")[-1].strip() for o in pred.split("\n") if o != ""]
        assert num == len(answer), f"answer: {len(answer)}, num: {num}"
    except AssertionError as e:
        if len(answer) > num:
            answer = answer[:num]
        elif len(answer) < num:
            answer.extend(["not know"] * (num - len(answer)))
    return answer

def filter(data):
    last = select_data[data["qid"]]["last"]
    if data["qid"] in select_data2:
        key = "answer" if "answer" in select_data2[data["qid"]] else "sub_cluster_ids"
        if len(select_data2[data["qid"]][key]) == 0:
            return data["idx"] % 2 == 0
        relevant = select_data2[data["qid"]][key]
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


def get_prompt(data):
    qid = data["qid"]
    if input_type == "qa":
        question = "\n".join([f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["question"])])
    else:
        question = data["question"]
    prompt = PROMPT1.replace("[question]", question)
    return prompt

def post_process(out, qid):
    if input_type == "qa":
        out = parse(out, len(question_data[qid]["question"]))
    return out

def frame_select(runner, **data):
    qid = data["qid"]
    prompt = get_prompt(data)
    try:
        out = model.forward(prompt, data["frame"])
        out = post_process(out, qid)
        out = {"answer": out, "qid": qid, "idx": data["idx"], "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out

def frame_select2(runner, batch_data):
    batch_prompt, batch_frame, batch_out = [], [], []
    for data in batch_data:
        prompt = get_prompt(data)
        batch_prompt.append(prompt)
        batch_frame.append(data["frame"])
    try:
        out = model.forward(batch_prompt, batch_frame)
    except Exception as e:
        print(e)
        out = None
    for data, o, prompt in zip(batch_data, out, batch_prompt):
        qid = data["qid"]
        o = post_process(o, qid)
        batch_out.append({"answer": o, "qid": qid, "idx": data["idx"], "prompt": prompt})
    return batch_out

async def async_frame_select(runner, **data):
    qid = data["qid"]
    prompt = get_prompt(data)
    try:
        out = await model.forward(prompt, data["frame"])
        out = post_process(out, qid)
        out = {"answer": out, "qid": qid, "idx": data["idx"], "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    exp_name = "0512"
    select_data = load_data("./outputs/0413/select.jsonl")
    question_data = load_data("./outputs/0404/question2.jsonl")
    
    batch_size = 8
    use_api = True
    uniform_sample = False
    # input_type = "qa"
    input_type = "analysis"
    select_type = "tree"
    
    model_name = "gpt-4o-2024-05-13"
    output_path = f"./outputs/{exp_name}/answer2_{model_name}_{input_type}.jsonl"
    
    if select_type == "tree":
        select_data2 = load_data("./outputs/0329/nextmc_gpt_4o_tree2.jsonl")
    elif select_type == "wo_capiton":
        select_data2 = load_data("./outputs/0413/select2.jsonl")
    elif select_type == "human":
        select_data2 = load_data("./outputs/0502/relevant.jsonl")
        
    model = create_model("api", model_name)
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
