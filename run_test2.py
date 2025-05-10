from dataset.builder import build_dataset
from args import TestingArguments
from transformers import HfArgumentParser
from pathlib import Path
import decord

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from utils import load_data
from task_utils import api_forward, list2dict
import asyncio
import jsonlines
import json
from tasks import Description

load_dotenv()
client = OpenAI()

exp_name = "0323_desc"
task_type = Description()

async def run_sigle(total, num, text, frame, qid, idx, sem):
    async with sem:
        try:
            out = await api_forward(text, frame)
            if out is not None:
                out = task_type.parse2(out, num)
                return {"pred": out, "qid": qid, "idx": idx, "total": total, "prompt": text}
            return None
        except Exception as e:
            print(e)
            return None

def load_jsonl2dict(path):
    list_data = jsonlines.open(path, "r")
    from collections import defaultdict
    processed = defaultdict(dict)
    for data in list_data:
        processed[data["qid"]][data["idx"]] = data
    return processed

async def eval():
    args = HfArgumentParser(TestingArguments).parse_args_into_dataclasses()[0]

    eval_dataset = build_dataset(
        args.dataset_config, args.eval_dataset, is_training=False
    )
    output_path = f"./outputs/{exp_name}/nextqa2.jsonl"
    input_path = f"./outputs/{exp_name}/nextqa.jsonl"
    video_path = "D:/datasets/nextqa/NExTVideo"
    
    with jsonlines.open(output_path, "a") as writer:
        for dataset in eval_dataset.values():
            
            if Path(output_path).exists():
                processed = load_jsonl2dict(output_path)
            else:
                processed = {}
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).touch()
            input_data = load_data(input_path)
            tasks = []
            sem = asyncio.Semaphore(200)
            for data in dataset:
                if data["qid"] not in input_data:
                    continue
                pred = input_data[data["qid"]]["pred"]
                _video_path = Path(video_path).joinpath(data["video_path"])
                # scenes = detect(str(video_path), ContentDetector())
                vr = decord.VideoReader(str(_video_path))

                fps = vr.get_avg_fps()
                sub_task = []
                frame_idx = []
                for idx in range(0, len(vr), int(fps * 1)):
                    if data["qid"] in processed and idx in processed[data["qid"]]:
                        continue
                    frame_idx.append(idx)
                # if frame_idx[-1] != len(vr) - 1 and (
                # data["qid"] not in processed
                # or len(vr) - 1 not in processed[data["qid"]]
                # ):
                # frame_idx.append(len(vr) - 1)

                video = vr.get_batch(frame_idx)
                for frame, idx in zip(video, frame_idx):
                    parse_map = {
                        list: lambda x: "\n".join(x),
                        dict: lambda x: json.dumps(x),
                        str: lambda x: x,
                    }
                    text = task_type.prompt2().replace("[question]", parse_map[type(pred)](pred))
                    sub_task.append((text, frame, data["qid"], idx, sem))
                tasks.extend([run_sigle(len(sub_task), len(pred), *task) for task in sub_task])
            total, invalid = len(tasks), 0
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                out = await task
                if out is None:
                    invalid += 1
                    continue
                writer.write(out)

            print(f"Invalid: {invalid}/{total}")


if __name__ == "__main__":
    asyncio.run(eval())
