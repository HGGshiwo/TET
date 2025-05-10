""" 
1. 让模型从左到右阅读表格，如果可以直接回答，则忽略剩余的内容
2. 使用拼接的图片作为输入
"""
# 模型根据问题和回答选择视频帧
from runner import AsyncRunner
from pathlib import Path

import json
import asyncio
from utils import load_data
from task_utils import parse_json, api_forward, get_frame, make_grid

example = {
    "answer": "A",
    "explain": "put your explaination here",
    "confidence": 3,
}

PROMPT = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

async def frame_select(runner, **data):
    qid = data["qid"]
    valid = []
    if data["qid"] in select_data2:
        if len(select_data2[data["qid"]]["answer"]) != 0:
            valid = select_data2[data["qid"]]["answer"]
    elif data["qid"] in select_data1:
        if len(select_data1[data["qid"]]["relevant_idx"]) != 0:
            valid = select_data1[data["qid"]]["relevant_idx"]
    
    video_path = runner.dataset.config.video_path
    video_path = Path(video_path).joinpath(data["video_path"])
    frames = get_frame(video_path, 1)
    frames = [frames[i] for i in valid]
    image = make_grid(frames)
    question = data["question"]
    prompt = PROMPT.replace("[question]", question)
    try:
        out = await api_forward(prompt, image)
        image.save(f"./outputs/0504/{qid}.jpg")
        out = parse_json(out)
        if out:
            out["qid"] = qid
            out["prompt"] = prompt
            out["truth"] = data["truth"]
        else:
            out = None
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    output_path = "./outputs/0504/answer3_2.jsonl"
    select_data1 = load_data("./outputs/0413/select.jsonl")
    select_data2 = load_data("./outputs/0502/relevant.jsonl")
    
    runner = AsyncRunner(frame_select, output_path, iter_key="qid")
    asyncio.run(runner())
    
    
    # compute metrics
    result = load_data(output_path)
    compute_metrics = runner.dataset.get_compute_metrics2()
    for item in runner.dataset:
        if item["qid"] not in result:
            continue
        out = compute_metrics(result[item["qid"]]["answer"], item, True)
    print(out)