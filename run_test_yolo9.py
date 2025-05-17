""" 
1. 让模型从左到右阅读表格，如果可以直接回答，则忽略剩余的内容
"""
from runner import AsyncRunner
import decord

decord.bridge.set_bridge("torch")
import json
import asyncio
from utils import load_data
from task_utils import parse_json, create_model, generate_table, get_frame, make_grid
from pathlib import Path
import numpy as np
from datetime import datetime

example = {
    "answer": "A",
    "explain": "put your explaination here",
    "confidence": 3,
}


PROMPT1 = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT2 = f"This is a question related to the video: [question]. Here is a markdown table:\n[table]\n Each column represents an analysis to the question corresponding to the frame number at that time. Answer the question based on the information in the table. Read the table from left to right. If you find that what you have read in the table can answer the question, ignore the rest of the table and answer directly. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT3 = f"Here is a markdown table:\n[table]\nEach row represents a question, and each column represents the answer to the question corresponding to the frame number at that time. Answer the following question based on the information in the table: [question]. First, eliminate the wrong options through the table, and then choose the one you think is most appropriate from the remaining options. The time points in the table have been adjusted to the time corresponding to the question, and you can ignore the time information in the question. For example, for the question 'what does the lady do after shaking her body for a while in the middle of the video', you can ignore the time information: 'lady shaking her body' and 'in the middle of the video'. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT_TREE = f"This is a question related to the video: [question]. You are given some language descriptions of a video. The descriptions are sparsely sampled from the videos. Each description is preceded by a corresponding frame number. Here are the descriptions:\n[narration]\n Try to answer the question based on the descriptions. If you are not sure, answer with the most likely answer. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT_TREE2 = "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are not sure, answer with the most likely answer. You are given some language descriptions of a video. The descriptions are sparsely sampled from the videos. The description consists of several video-related questions and their corresponding answers, starting with the question and then the corresponding answer. Each answer is preceded by a corresponding frame number. \nHere are the descriptions:\n[narration]\n Here is the question: [question]"

def frame_filter(data):
    if use_difficult:
        data = difficult_data[data["qid"]]
        if data["answer"] != data["truth"]:
            return True
        return False
    return True

async def frame_select(runner, **data):
    qid = data["qid"]
    if select_type == "tree":
        idx = select_data2[data["qid"]]["sub_cluster_ids"]
        valid = list(set(idx))
    elif select_type == "human":
        valid = select_data2[data["qid"]]["answer"]
    elif select_type == "wo_capiton":
        # valid = [key for key in answer_data[qid].keys() if key % 2 == 0]
        valid = list(range(0, select_data1[data["qid"]]["last"], 2))
        if data["qid"] in select_data2:
            if len(select_data2[data["qid"]]["answer"]) != 0:
                valid = select_data2[data["qid"]]["answer"]
        elif data["qid"] in select_data1:
            if len(select_data1[data["qid"]]["relevant_idx"]) != 0:
                valid = select_data1[data["qid"]]["relevant_idx"]
    if uniform_sample:
        last = select_data1[data["qid"]]["last"]
        valid = np.linspace(0, last, len(valid)).astype(int).tolist()
    
    image = None
    if input_type == "qa":
        table = generate_table(question_data[qid]["question"], answer_data[qid])
        prompt = PROMPT3.replace("[table]", table)
    elif input_type == "analysis":
        answer = {key: dict(answer=[answer_data[qid][key]["answer"]]) for key in valid}
        table = generate_table(["Analysis"], answer)
        prompt = PROMPT2.replace("[table]", table)
    elif input_type == "caption":
        cur_narr = narr_data[data["vid"]]
        cur_narr = [cur_narr[k]["caption"] for k in sorted(cur_narr.keys())]
        captions = [f"{i}: " + cur_narr[i] for i in valid if i < len(cur_narr)]
        captions = "\n".join(captions)
        prompt = PROMPT_TREE.replace("[narration]", captions)
    elif input_type == "image":
        video_path = runner.dataset.config.video_path
        video_path = Path(video_path).joinpath(data["video_path"])
        frames = get_frame(video_path, 1)
        frames = [frames[i] for i in valid]
        image = make_grid(frames)
        image.save(f"./outputs/0504/{qid}.jpg")
        prompt = PROMPT1
    question = data["question"]
    prompt = prompt.replace("[question]", question)
    
    try:
        out = await model.forward(prompt, image)
        assert out is not None, "model output is None"
        out = parse_json(out)
        if not out:
            out = None
            print("model output is None")
        else:
            out["qid"] = qid
            out["prompt"] = prompt
            out["truth"] = data["truth"]
    except Exception as e:
        print(e)
        out = None
    return out


if __name__ == "__main__":
    exp_name = "0515"
    
    uniform_sample = False # 均匀采样，消融实验
    use_difficult = True # 使用困难样本
    
    # select_type = "tree" # video tree采样
    # select_type = "human" # 人工选择
    select_type = "wo_capiton" # 大模型选择
    
    # input_type = "image" # 使用拼接的图片输入
    # input_type = "qa" # 使用问题和回答输入
    # input_type = "analysis" # 使用分析表格输入
    input_type = "caption" # 使用视频描述输入
    
    # model_name = "gpt-4o-2024-05-13"
    # model_name = 'gpt-4.1-2025-04-14'
    model_name = "qwen-vl-max"
    # model_name = "gpt-4o"
    
    output_path = f"./outputs/{exp_name}/answer3_{select_type}_{input_type}_{model_name}.jsonl"
    # output_path = "./outputs/0329/nextmc_gpt_4o_tree3_explain.jsonl"
    question_data = load_data("./outputs/0404/question2.jsonl")
    select_data1 = load_data("./outputs/0413/select.jsonl")
    difficult_data = load_data("./outputs/0510/filter.jsonl")
    
    if select_type == "tree":
        select_data2 = load_data("./outputs/0329/nextmc_gpt_4o_tree2.jsonl")
    elif select_type == "wo_capiton":
        select_data2 = load_data("./outputs/0413/select2.jsonl")
    elif select_type == "human":
        select_data2 = load_data("./outputs/0502/relevant.jsonl")
        
    if input_type == "analysis":
        answer_data = load_data("./outputs/0512/answer2_gpt-4o-2024-05-13_analysis.jsonl")
    
    if input_type == "caption":
        narr_path = "./outputs/0329/nextmc_gpt_4o.jsonl"
        narr_data = load_data(narr_path)
        
    model = create_model("api", model_name)
    
    runner = AsyncRunner(frame_select, output_path, filter=frame_filter, iter_key="qid")
    asyncio.run(runner())
    
    # compute metrics
    result = load_data(output_path)
    compute_metrics = runner.dataset.get_compute_metrics2()
    total, difficult = 0, 0
    for item in runner.dataset:
        total += 1
        if difficult_data[item["qid"]]["answer"] != difficult_data[item["qid"]]["truth"]:
            difficult += 1
        if item["qid"] not in result:
            continue
        if not frame_filter(item):
            continue
        if "pred" in result[item["qid"]]:
            out = compute_metrics(result[item["qid"]]["pred"], item, True)
        else:
            out = compute_metrics(result[item["qid"]]["answer"], item, True)
    print(out)
    if use_difficult:
        print(f"difficult rate: {difficult / total}[{difficult}/{total}]")