# 2. 给出起始帧描述，然后让模型给出问题
from runner import AsyncRunner

import json
import asyncio
from task_utils import parse_json, api_forward

example = ["woman in blue", "boy"]

PROMPT = f"Here is a video-related question: [quesiton]. Extract possible objects that may appear in the scene from the question and options. Output a json list, each item in the list is the name of the object. Output example: {json.dumps(example)}"


async def frame_select(runner, **data):
    qid = data["qid"]
    question = data["question"]
    prompt = PROMPT.replace("[quesiton]", question)
    try:
        out = await api_forward(prompt)
        if "[" in out:
            out = parse_json(out, list=True)
            out = {"qid": qid, "pred": out, "obj": out}
        else:
            out = {"qid": qid, "pred": out, "obj": None}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0508/object.jsonl"
    runner = AsyncRunner(frame_select, output_path, iter_key="qid")
    asyncio.run(runner())
