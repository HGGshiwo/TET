""" 
1. 让模型从左到右阅读表格，如果可以直接回答，则忽略剩余的内容
2. 消融实验, 
"""
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

PROMPT = f"This is a question related to the video: [question]. Here is a markdown table:\n[table]\n Each column represents an analysis to the question corresponding to the frame number at that time. Answer the question based on the information in the table. Read the table from left to right. If you find that what you have read in the table can answer the question, ignore the rest of the table and answer directly. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

async def frame_select(**data):
    qid = data["qid"]
    answer = {key: dict(answer=[value["answer"]]) for key, value in answer_data[qid].items()}
    table = generate_table(["Analysis"], answer)
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
    output_path = "./outputs/0416/answer3_xr1.jsonl"
    answer_data = load_data("./outputs/0416/answer2_xr1.jsonl")
    runner = AsyncRunner(frame_select, output_path, iter_key="qid")
    asyncio.run(runner())
    
    # compute metrics
    result = load_data(output_path)
    compute_metrics = runner.dataset.get_compute_metrics2()
    for item in runner.dataset:
        out = compute_metrics(result[item["qid"]]["answer"], item, True)
    print(out)