# 2. 给出起始帧描述，然后让模型给出问题
from runner import AsyncRunner
from task_utils import create_model

import json
import asyncio
from task_utils import parse_json

example = {
    "phrase": ["boy lost red toy"],
    "quesiton": ["Does the boy hold the red toy?"],
}

PROMPT = f"Here is a video-related question: [quesiton]. Try to extract subject-predicate phrases contained in the question. Here is an example: 'how many children are in the video' does not contain any phrases, while 'why is the blue sweater guy looking at the shirtless men' contains one phrase, namely blue sweater guy looking at the shirtless men. If the number of phrase is not zero, please design several questions for each frame to test whether the frame matches the phrase. If there is zero event in the original question, output No, then explain the reason, otherwise output a json. The keys of json are 'phrase' and 'question', and the values are phrase mentioned in the question and the new designed question, separately. New designed question must be a list, and each value in the list represents a question. There should be no more than 3 questions, and do not use pronouns in each question, but use full names. Keep your questions short and simple, and don't include questions with similar meanings. Output example: {json.dumps(example)}"


async def frame_select(runner, **data):
    qid = data["qid"]
    question = data["question"].split("A. ")[0].strip()
    prompt = PROMPT.replace("[quesiton]", question)
    try:
        out = await model.forward(prompt)
        if "{" in out:
            out = parse_json(out)
            out["qid"] = qid
        else:
            out = {"qid": qid, "event": None, "question": None, "pred": out}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    # exp_name = "0404"
    exp_name = "0522"
    
    model_name = "gpt-4o"
    
    # dataset_name = "nextmc_test"
    dataset_name = "egoschema_subset"
    
    model = create_model('api', model_name)
    output_path = f"./outputs/{exp_name}/question_{model_name}.jsonl"
    runner = AsyncRunner(frame_select, output_path, iter_key="qid", dataset=dataset_name)
    asyncio.run(runner())
