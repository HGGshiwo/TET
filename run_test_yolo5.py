# 2. 给出起始帧描述，然后让模型给出问题
from runner import Runner, AsyncRunner
from utils import load_data
from task_utils import create_model
import asyncio

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
    if (
        question_data[data["qid"]]["question"] is not None
        and data["idx"] in select_data[data["qid"]]["relevant_idx"]
    ):
        return True
    return False


def frame_select(runner, **data):
    qid = data["qid"]
    question = "\n".join(
        [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["question"])]
    )
    prompt = PROMPT.replace("[question]", question)
    try:
        out = model.forward(prompt, data["frame"])
        out = parse(out, len(question_data[qid]["question"]))
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
            [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["question"])]
        )
        prompt = PROMPT.replace("[question]", question)
        batch_prompt.append(prompt)
        batch_frame.append(data["frame"])
    
    batch_out = model.forward(batch_prompt, batch_frame)
    batch_out2 = []
    for data, out in zip(batch_data, batch_out):
        qid = data["qid"]
        try:
            out = parse(out, len(question_data[qid]["question"]))
            out = {"answer": out, "qid": qid, "idx": data["idx"]}
        except Exception as e:
            print(e)
            out = None
        batch_out2.append(out)
    return batch_out2

async def async_frame_select(runner, **data):
    qid = data["qid"]
    question = "\n".join(
        [f"{i+1}: {q}" for i, q in enumerate(question_data[qid]["question"])]
    )
    prompt = PROMPT.replace("[question]", question)
    try:
        out = await model.forward(prompt, data["frame"])
        out = parse(out, len(question_data[qid]["question"]))
        out = {"answer": out, "qid": qid, "idx": data["idx"]}
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    
    model_name = "gpt-4o"
    # model_name = "qwen"
    # model_name = "llava"
    
    # exp_name = "0420"
    exp_name = "0522"
    
    output_path = f"./outputs/{exp_name}/answer_{model_name}.jsonl"
    
    # select_data = load_data("./outputs/0413/select.jsonl")
    select_data = load_data("./outputs/0522/select.jsonl")
    
    # question_data = load_data("./outputs/0404/question.jsonl")
    question_data = load_data("./outputs/0522/question_gpt-4o.jsonl")
    
    # dataset_name = "nextmc_test"
    dataset_name = "egoschema_subset"
    
    use_api = model_name in ["gpt-4o"]
    kwargs = {}
    if use_api:
        model = create_model("api", model_name)
    else:
        model = create_model(model_name)
        kwargs = {"batch_size": 8}    
        
    task_func = None
    if use_api:
        task_func = async_frame_select 
    else:
        if kwargs.get("batch_size", 1) == 1:
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
        dataset=dataset_name,
        **kwargs,
    )
    if use_api:
        asyncio.run(runner())
    else:
        runner()
