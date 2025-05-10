""" 
1. 模型回答设计的问题
"""
import asyncio
from utils import load_data
from task_utils import api_forward
from runner import AsyncRunner, Runner

PROMPT = "Please provide a brief answer to the following questions: [question], output the answer directly without other text, if there is not enough information in the frame, just answer 'not know', seperate an with line breaks. Output Example:\n1: answer1\n2: answer2\n3: answer3"


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
    if data["qid"] in select_data2:
        return data["idx"] in select_data2[data["qid"]]["answer"]
    elif data["qid"] in select_data:
        return data["idx"] in select_data[data["qid"]]["relevant_idx"]
    return False


def frame_select(**data):
    qid = data["qid"]
    question = "\n".join(
        [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["question"])]
    )
    prompt = PROMPT.replace("[question]", question)
    try:
        out = model.forward(prompt, data["frame"])
        out = parse(out, len(question_data[qid]["question"]))
        out = {"answer": out, "qid": qid, "idx": data["idx"], "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out


async def async_frame_select(**data):
    qid = data["qid"]
    question = "\n".join(
        [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["question"])]
    )
    prompt = PROMPT.replace("[question]", question)
    try:
        out = await api_forward(prompt, data["frame"])
        out = parse(out, len(question_data[qid]["question"]))
        out = {"answer": out, "qid": qid, "idx": data["idx"], "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0404/answer2.jsonl"
    select_data = load_data("./outputs/0404/select.jsonl")
    select_data2 = load_data("./outputs/0404/select2.jsonl")
    question_data = load_data("./outputs/0404/question2.jsonl")
    use_api = True

    if not use_api:
        model = load_llava_video()

    task_func = async_frame_select if use_api else frame_select
    runner_cls = AsyncRunner if use_api else Runner
    runner = runner_cls(
        task_func,
        output_path,
        iter_key="qid",
        iter_frame=True,
        video_fps=1,
        filter=filter,
    )

    if not use_api:
        runner()
    else:
        asyncio.run(runner())
