# 让模型给出问题
from runner import AsyncRunner
import asyncio
from task_utils import api_forward

PROMPT = "This is a video-related question: [question], please design several questions, which can be a request to describe the content of the picture or a question about the content of the picture. For example, the new designed question can be: 'describe the object in the woman's hand' or 'Is the woman holding the red ball?'. The new designed questions must be used to eliminate wrong options or provide information for choosing the correct options. If the original question contains time-related conditions such as after, while, etc., please ignore them. For example, When designing questions for 'what does the lady do after shaking her body for a while', you don't need to consider the condition 'woman shaking her body for a while', only need to consider the subject of the problem 'what is the lady doing'. Split the new designed questions with line breaks, do not output extra text. The number of questions should be less than 5. Output example:\nquestion1\nquestion2\nquestion3"


async def frame_select(**data):
    question = data["question"]
    prompt = PROMPT.replace("[question]", question)
    try:
        pred = await api_forward(prompt)
        out = [out for out in pred.split("\n") if out.strip() != ""]
        out = {"qid": data["qid"], "question": out}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0404/question2.jsonl"
    runner = AsyncRunner(frame_select, output_path, iter_key="qid")
    asyncio.run(runner())
