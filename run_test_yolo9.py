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
from task_utils import crop_img, make_crop_grid

example = {
    "answer": "A",
    "explain": "put your explaination here",
    "confidence": 3,
}

PROMPT1 = f"This is a question related to the video: [question]. Here are the frames related to the question. The image is composed of several frames stitched together in chronological order, with each frame separated by a black border. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Try to answer the questions based on the information in the picture. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT1_2 = "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are not sure, answer with the most likely answer. You are given a image  composed of several frames stitched together in chronological order, with each frame separated by a black border. The frames are sparsely sampled from the videos. The frames in each row increase in time from left to right, and the first frame of the next row follows immediately after the last frame of the previous row. Here is the question: [question]"

PROMPT2 = f"This is a question related to the video: [question]. Here is a markdown table:\n[table]\n Each column represents an analysis to the question corresponding to the frame number at that time. Answer the question based on the information in the table. Read the table from left to right. If you find that what you have read in the table can answer the question, ignore the rest of the table and answer directly. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT2_2 = "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are not sure, answer with the most likely answer. You are given a markdown table:\n[table]\n Each column represents an analysis to the question corresponding to the frame number at that time.Here is the question: [question]"

PROMPT3 = f"Here is a markdown table:\n[table]\nEach row represents a question, and each column represents the answer to the question corresponding to the frame number at that time. Answer the following question based on the information in the table: [question]. First, eliminate the wrong options through the table, and then choose the one you think is most appropriate from the remaining options. The time points in the table have been adjusted to the time corresponding to the question, and you can ignore the time information in the question. For example, for the question 'what does the lady do after shaking her body for a while in the middle of the video', you can ignore the time information: 'lady shaking her body' and 'in the middle of the video'. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT_TREE = f"This is a question related to the video: [question]. You are given some language descriptions of a video. The descriptions are sparsely sampled from the videos. Each description is preceded by a corresponding frame number. Here are the descriptions:\n[narration]\n Try to answer the question based on the descriptions. If you are not sure, answer with the most likely answer. Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

PROMPT_TREE2 = "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are not sure, answer with the most likely answer. You are given some language descriptions of a video. The descriptions are sparsely sampled from the videos. The description consists of several video-related questions and their corresponding answers, starting with the question and then the corresponding answer. Each answer is preceded by a corresponding frame number. \nHere are the descriptions:\n[narration]\n Here is the question: [question]"

PROMPT4 = f"Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are not sure, answer with the most likely answer. Output a json string with following keys: 'answer', 'confidence', 'explain'. You are given some language descriptions of a video. The descriptions are sparsely sampled from the videos. The description consists of several video-related questions and their corresponding answers, starting with the question and then the corresponding answer. Each answer is preceded by a corresponding frame number.\nHere are the descriptions:[description]\nHere is the question: [question], Output example: {json.dumps(example)}"

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
    elif select_type == "dino":
        valid = select_data2[data["qid"]]["relevant_idx"]
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
        if select_type == "dino":
            last = results_data[data["qid"]]["last"]
        else:
            last = select_data1[data["qid"]]["last"]
        valid = np.linspace(0, last-1, len(valid)).astype(int).tolist()
        valid = list(set(valid))
        
    image = None
    if input_type == "qa":
        table = generate_table(question_data[qid]["question"], answer_data[qid])
        prompt = PROMPT3.replace("[table]", table)
    elif input_type == "analysis" or input_type == "analysis-old":
        answer = {key: dict(answer=[answer_data[qid][key]["answer"]]) for key in valid}
        table = generate_table(["Analysis"], answer)
        prompt_base = PROMPT2 if input_type == "analysis" else PROMPT2_2
        prompt = prompt_base.replace("[table]", table)
    elif input_type == "caption" or input_type == "caption-old":
        cur_narr = narr_data[data["vid"]]
        cur_narr = [cur_narr[k]["caption"] for k in sorted(cur_narr.keys())]
        captions = [f"{i}: " + cur_narr[i] for i in valid if i < len(cur_narr)]
        captions = "\n".join(captions)
        base_prompt = PROMPT_TREE if input_type == "caption" else PROMPT_TREE2
        prompt = base_prompt.replace("[narration]", captions)
    elif input_type == "image" or input_type == "image-old":
        video_path = runner.dataset.config.video_path
        video_path = Path(video_path).joinpath(data["video_path"])
        frames = get_frame(video_path, 1)
        if use_crop and (results := results_data[qid]["results"]):
            boxes = []
            for i in valid:
                if str(i) not in results:
                    continue
                boxes.append([b for v in results[str(i)].values() for b in v["boxes"]])
            image = make_crop_grid([frames[i] for i in valid], boxes, max_frame=max_frame)
        else:
            frames = [frames[i] for i in valid]
            image = make_grid(frames, max_frame)
        save_dir = Path(output_path.replace(".jsonl", "_image"))
        save_dir.mkdir(parents=True, exist_ok=True)
        image.save(str(save_dir.joinpath(f"{qid}.jpg")))
        prompt = PROMPT1 if input_type == "image" else PROMPT1_2
    question = data["question"]
    prompt = prompt.replace("[question]", question)
    
    try:
        if use_old_input:
            prompt = input_data[qid]["prompt"]    
        out = await model.forward(prompt, image)
        assert out is not None, "model output is None"
        if use_old_input or input_type.endswith("-old"):
            out = {"answer": out}
        else:
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
    # exp_name = "0515"
    # exp_name = "0522"
    # exp_name = "0601"
    # exp_name = "0603"
    # exp_name = "0606" # wo skip
    # exp_name = "0607"
    # exp_name = "0609"
    # exp_name = "0610"
    # exp_name = "0613"
    exp_name =  "0621"
    
    # dataset_name = "egoschema_subset"
    dataset_name = "nextmc_test"
    
    uniform_sample = False # 均匀采样，消融实验
    # uniform_sample = True
    
    use_difficult = False # 使用困难样本
    use_old_input = False # 使用之前的输入
    use_crop = True # 是否裁剪图片
    # use_crop = False # 是否裁剪图片
        
    # select_type = "tree" # video tree采样
    # select_type = "human" # 人工选择
    # select_type = "wo_capiton" # 大模型选择
    select_type = "dino"
    
    # max_frame = 16
    # max_frame = 8 # 只使用前8帧
    # max_frame = 24
    max_frame = 48
    
    input_type = "image" # 使用拼接的图片输入
    # input_type = "qa" # 使用问题和回答输入
    # input_type = "analysis" # 使用分析表格输入
    # input_type = "analysis-old" # 使用分析表格输入
    # input_type = "caption" # 使用视频描述输入
    # input_type = "caption-old" # 使用视频描述输入
    # input_type = "image-old"
    
    # model_name = "gpt-4o-2024-05-13"
    model_name = 'gpt-4.1-2025-04-14'
    # model_name = "qwen-vl-max"
    # model_name = "gpt-4o"
    
    if use_crop or (select_type == "dino" and uniform_sample):
        # results_data = load_data("./outputs/0607/dino_out_nextmc_test.jsonl")
        results_data = load_data("./outputs/0619/dino_out_nextmc_test_low_tiny.jsonl")
        
    end = "" if not use_crop else "_crop"
    end += "" if not uniform_sample else "_uniform"
    output_path = f"./outputs/{exp_name}/answer3_{select_type}_{input_type}_{model_name}_{max_frame}{end}.jsonl"
    # output_path = "./outputs/0329/nextmc_gpt_4o_tree3_explain.jsonl"
    
    if input_type == "qa":
        question_data = load_data("./outputs/0404/question2.jsonl")
    
    # select_data1 = load_data("./outputs/0413/select.jsonl")
    select_data1 = load_data("./outputs/0522/select.jsonl")
    
    if use_difficult:
        if dataset_name == "nextmc_test":
            difficult_data = load_data("./outputs/0510/filter.jsonl")
        elif dataset_name == "egoschema_subset":
            difficult_data = load_data("./outputs/0522/filter_gpt-4o_egoschema_subset.jsonl")
    
    if select_type == "tree":
        select_data2 = load_data("./outputs/0329/nextmc_gpt_4o_tree2.jsonl")
    elif select_type == "wo_capiton":
        # select_data2 = load_data("./outputs/0413/select2.jsonl")
        select_data2 = load_data("./outputs/0522/select2_gpt-4o.jsonl")
    elif select_type == "human":
        select_data2 = load_data("./outputs/0502/relevant.jsonl")
    elif select_type == "dino":
        # select_data2 = load_data("./outputs/0601/dino_select.jsonl")
        # select_data2 = load_data("./outputs/0604/dino_select_skip.jsonl")
        # select_data2 = load_data("./outputs/0607/dino_select_gpt-4.1_nextmc_test.jsonl")
        # select_data2 = load_data("./outputs/0607/dino_select_gpt-4.1_egoschema_subset.jsonl")
        # select_data2 = load_data("./outputs/0620/dino_select_gpt-4.1_nextmc_test.jsonl")
        select_data2 = load_data("./outputs/0621/dino_select_gpt-4.1_nextmc_test.jsonl")
    if input_type == "analysis" or input_type == "analysis-old":
        answer_data = load_data("./outputs/0512/answer2_gpt-4o-2024-05-13_analysis.jsonl")
    
    if input_type == "caption" or input_type == "caption-old":
        narr_path = "./outputs/0329/nextmc_gpt_4o.jsonl"
        narr_data = load_data(narr_path)
    
    if use_old_input:
        input_data = load_data("./outputs/0329/nextmc_gpt_4o_tree3_explain.jsonl")
        output_path = f"./outputs/{exp_name}/old_input_{model_name}.jsonl"
        
    model = create_model("api", model_name)
    
    runner = AsyncRunner(frame_select, output_path, filter=frame_filter, iter_key="qid", dataset=dataset_name)
    asyncio.run(runner())
    
    # compute metrics
    result = load_data(output_path)
    compute_metrics = runner.dataset.get_compute_metrics2()
    total, difficult = 0, 0
    for item in runner.dataset:
        total += 1
        if use_difficult:
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
    failed = out.pop("failed")
    failed_path = output_path.replace(".jsonl", ".txt")
    Path(failed_path).write_text("\n".join(failed))
    print(out)
    if use_difficult:
        print(f"difficult rate: {difficult / total}[{difficult}/{total}]")
