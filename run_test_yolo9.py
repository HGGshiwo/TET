# 模型根据问题和回答选择视频帧
from runner import AsyncRunner
import decord

decord.bridge.set_bridge("torch")
import json
import asyncio
from utils import load_data
from task_utils import parse_json, api_forward, generate_table

example = {
    "answer": "A",
    "explain": "put your explaination here",
    "confidence": 3,
}

PROMPT = f"Here is a markdown table:\n[table]\nEach row represents a question, and each column represents the answer to the question corresponding to the frame number at that time. Answer the following question based on the information in the table: [question]. First, eliminate the wrong options through the table, and then choose the one you think is most appropriate from the remaining options. The time points in the table have been adjusted to the time corresponding to the question, and you can ignore the time information in the question. For example, for the question 'what does the lady do after shaking her body for a while in the middle of the video', you can ignore the time information: 'lady shaking her body' and 'in the middle of the video'. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"


async def frame_select(**data):
    qid = data["qid"]
    table = generate_table(question_data[qid]["question"], answer_data[qid])
    prompt = PROMPT.replace("[table]", table)
    question = data["question"]
    prompt = prompt.replace("[question]", question)
    
    try:
        out = await api_forward(prompt)
        out = parse_json(out)
        out["qid"] = qid
        out["prompt"] = prompt
        out["truth"] = data["truth"]
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0404/answer3.jsonl"
    question_data = load_data("./outputs/0404/question2.jsonl")
    answer_data = load_data("./outputs/0404/answer2.jsonl")
    runner = AsyncRunner(frame_select, output_path, iter_key="qid")
    asyncio.run(runner())
    
    # compute metrics
    result = load_data(output_path)
    compute_metrics = runner.dataset.get_compute_metrics2()
    for item in runner.dataset:
        out = compute_metrics(result[item["qid"]]["answer"], item, True)
    print(out)