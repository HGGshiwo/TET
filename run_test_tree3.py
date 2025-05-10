from dataset.builder import build_dataset
from args import GeneratingArguments
from transformers import HfArgumentParser
from pathlib import Path
import decord

decord.bridge.set_bridge("torch")
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from utils import load_data
from task_utils import api_forward
import asyncio
import jsonlines
import decord
from run_test2 import load_jsonl2dict
import json
from tasks import Description

load_dotenv()
client = OpenAI()


async def run_task(data, prompt, sem):
    async with sem:
        try:
            out = await api_forward(prompt)
            return data, {"qid": data["qid"], "pred": out, "prompt": prompt}
        except Exception as e:
            print(e)
            return data, None


video_path = "D:/datasets/nextqa/NExTVideo"
exp_name = "0329"
task_type = Description()

async def eval():
    args = HfArgumentParser(GeneratingArguments).parse_args_into_dataclasses()[0]

    dataset = build_dataset(args.dataset_config, args.dataset_name, is_training=False)
    enhance = False
    explain = True
    prompt_explain = "After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer." if explain else "You must not provide any other response or explanation"
    
    input_path = f"./outputs/{exp_name}/nextmc_gpt_4o_tree2.jsonl"
    name2 = "_explain" if explain else ""
    name2 = name2 + "_enhance" if enhance else name2
    output_path = f"./outputs/{exp_name}/nextmc_gpt_4o_tree3{name2}.jsonl"
    
    if not enhance:
        # narr_path = "D:/datasets/LLoVi_caption/nextqa/llava1.5_fps1.json"
        narr_path = "./outputs/0329/nextmc_gpt_4o.jsonl"
        narr = load_data(narr_path)
        PROMPT = f"Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E). {prompt_explain}. If you are not sure, answer with the most likely answer. {task_type.prompt3()}\nHere are the descriptions:\n$narration\n Here is the question: $question?\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n"
    else:
        narr_path = f"./outputs/{exp_name}/nextqa2.jsonl"
        narr = load_jsonl2dict(narr_path)
        data1 = load_data(f'./outputs/{exp_name}/nextqa.jsonl')
        PROMPT = f"Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the letters (A, B, C, D, or E).{prompt_explain}. If you are not sure, answer with the most likely answer. {task_type.prompt3()}.\nHere are the descriptions:\n$narration\n Here is the question: $question?\n Here are the choices:\n (A): $optionA\n (B): $optionB\n (C): $optionC\n (D): $optionD\n (E): $optionE\n"
    with jsonlines.open(output_path, "a") as writer:
        if Path(output_path).exists():
            processed = load_data(output_path)
        else:
            processed = {}
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).touch()
        input_data = load_data(input_path)
        compute_metrics = dataset.get_compute_metrics2()
        tasks = []
        sem = asyncio.Semaphore(200)
        
        for data in dataset:
            if data["qid"] not in input_data:
                continue
            if data["qid"] in processed:
                out = processed[data["qid"]]["pred"]
                metrics = compute_metrics(out, data, True)
                continue

            idx = input_data[data["qid"]]["sub_cluster_ids"]
            idx = list(set(idx))
            if enhance:
                _video_path = Path(video_path).joinpath(data["video_path"])
                # scenes = detect(str(video_path), ContentDetector())
                vr = decord.VideoReader(str(_video_path))

                fps = vr.get_avg_fps()
                frame_map = list(range(0, len(vr), int(fps * 1)))
                cur_narr = narr[data["qid"]]
                map_narr = [cur_narr[frame_map[i]] for i in range(len(list(cur_narr.values())))]
                captions = task_type.build3(data, data1[data['qid']], map_narr, idx)
            else:
                cur_narr = narr[data["vid"]]
                cur_narr = [cur_narr[k]["caption"] for k in sorted(cur_narr.keys())]
                captions = [f"{i}: " + cur_narr[i] for i in idx]
                captions = "\n".join(captions)

            prompt = PROMPT.replace("$narration", captions)
            prompt = prompt.replace("$question", data["question"].split("A. ")[0])
            prompt = prompt.replace("$optionA", data["cm_a0"])
            prompt = prompt.replace("$optionB", data["cm_a1"])
            prompt = prompt.replace("$optionC", data["cm_a2"])
            prompt = prompt.replace("$optionD", data["cm_a3"])
            prompt = prompt.replace("$optionE", data["cm_a4"])
            tasks.append(run_task(data, prompt, sem))
        total, invalid = len(tasks), 0
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            data, out = await task
            if out is not None and out["pred"] is not None:
                out["truth"] = data["truth"]
                writer.write(out)
                metrics = compute_metrics(out["pred"], data, True)
            else:
                invalid += 1
        print(f"Invalid: {invalid}/{total}")
        print(metrics)


if __name__ == "__main__":
    asyncio.run(eval())
