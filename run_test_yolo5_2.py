# 2. 给出起始帧描述，然后让模型给出问题
from runner import Runner, AsyncRunner
from utils import load_data
from task_utils import create_model, api_forward
import asyncio

PROMPT = "Briefly analyze whether the picture is relevant to the given phrase: [question], output the answer directly without other text, If there is more than one phrase, separate the analysis corresponding to the phrase with line breaks"


def parse(pred, num):
    answer = [o.split(":")[-1].strip() for o in pred.split("\n") if o != ""]
    assert num == len(answer), f"answer: {len(answer)}, num: {num}"
    return answer


def filter(data):
    if (
        question_data[data["qid"]]["question"] is not None
        and data["idx"] in select_data[data["qid"]]["relevant_idx"]
    ):
        return True
    return False


def frame_select(**data):
    qid = data["qid"]
    question = "\n".join(
        [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["phrase"])]
    )
    prompt = PROMPT.replace("[question]", question)
    try:
        out = model.forward(prompt, data["frame"])
        out = parse(out, len(question_data[qid]["phrase"]))
        out = {"answer": out, "qid": qid, "idx": data["idx"]}
    except Exception as e:
        print(e)
        out = None
    return out


def frame_select2(runner, batch_data):
    batch_prompt, batch_frame = [], []
    for data in batch_data:
        qid = data["qid"]
        question = "\n".join(
            [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["phrase"])]
        )
        prompt = PROMPT.replace("[question]", question)
        batch_prompt.append(prompt)
        batch_frame.append(data["frame"])
    
    batch_out = model.forward(batch_prompt, batch_frame)
    batch_out2 = []
    for data, out in zip(batch_data, batch_out):
        qid = data["qid"]
        try:
            out = parse(out, len(question_data[qid]["pharse"]))
            out = {"answer": out, "qid": qid, "idx": data["idx"]}
        except Exception as e:
            print(e)
            out = None
        batch_out2.append(out)
    return batch_out2

async def async_frame_select(runner, **data):
    qid = data["qid"]
    if isinstance(question_data[qid]["phrase"], str):
        question_data[qid]["phrase"] = [question_data[qid]["phrase"]]
    question = "\n".join(
        [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["phrase"])]
    )
    prompt = PROMPT.replace("[question]", question)
    try:
        out = await api_forward(prompt, data["frame"])
        out = parse(out, len(question_data[qid]["phrase"]))
        out = {"answer": out, "qid": qid, "idx": data["idx"]}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    use_api = True
    use_qwen = True
    batch_size = 8
    output_path = "./outputs/0424/answer.jsonl"
    select_data = load_data("./outputs/0413/select.jsonl")
    question_data = load_data("./outputs/0404/question.jsonl")

    if not use_api:
        model_type = "qwen" if use_qwen else "llava"
        model = create_model(model_type)
        
    task_func = None
    if use_api:
        batch_size = 1
        task_func = async_frame_select 
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
    if use_api:
        asyncio.run(runner())
    else:
        runner()
