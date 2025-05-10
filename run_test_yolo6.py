# 模型根据问题和回答选择视频帧
from runner import AsyncRunner

import asyncio
from utils import load_data
from task_utils import parse_json, api_forward, generate_table

PROMPT = "Here is a markdown table:\n[table]\nEach row represents a question, and each column represents the answer to the question corresponding to the frame number at that time. According to the table, what is the time segment corresponding to question [question]? Note that if the question contains time-related words, you need to take this into consideration when selecting the segment. For example, if the question contains 'after' an event, you should select the segment after the event, not the segment where the event itself is located. If the original question says in the beginning/middle/end of the video, this refers to the first/middle/last 30% of the frames. Output a json string, with the keys 'explain' and 'time'. When the key is 'explain', the value is the reason for selecting the time period. When the key is 'time', the value is a list like string, which allows number or number-number to represent a specific frame number or the start to end of the frame number as the element of the list, for example '[0, 3-5, 7-8, 10]'. The number must range from 0 to [video_length]"


def parse(pred):
    pred = parse_json(pred)
    pred = pred["time"]
    out = []
    answer = pred.split("[")[-1].split("]")[0].strip()
    if answer == "":
        return out
    answer = answer.split(",")
    if len(answer) == 0:
        return out
    for ans in answer:
        ans = ans.strip()
        if "-" in ans:
            start, end = ans.split("-")
            out.extend(list(range(int(start), int(end) + 1)))
        else:
            out.append(int(ans))
    return out


def filter(data):
    if question_data[data["qid"]]["question"] is not None:
        return True
    return False


async def frame_select(runner, **data):
    qid = data["qid"]
    table = generate_table(question_data[qid]["question"], answer_data[data["qid"]])
    prompt = PROMPT.replace("[table]", table)
    question = data["question"].split("A. ")[0].strip()
    prompt = prompt.replace("[question]", question)
    prompt = prompt.replace("[video_length]", str(select_data[data["qid"]]["last"] - 1))
    try:
        out = await api_forward(prompt)
        ans = parse(out)
        out = {"answer": ans, "qid": qid, "pred": out, "prompt": prompt}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0420/select2.jsonl"
    select_data = load_data("./outputs/0413/select.jsonl")
    question_data = load_data("./outputs/0404/question.jsonl")
    answer_data = load_data("./outputs/0420/answer.jsonl")
    runner = AsyncRunner(
        frame_select,
        output_path,
        iter_key="qid",
        filter=filter,
    )
    asyncio.run(runner())
