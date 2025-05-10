from dataclasses import asdict
from model.builder import build_model
from dataset.data_collator import get_data_collator
from dataset.builder import build_dataset
from args import TestingArguments
from transformers import HfArgumentParser
from scenedetect import detect, ContentDetector
from pathlib import Path
import decord
import asyncio

decord.bridge.set_bridge("torch")
from tqdm import tqdm
from task_utils import api_forward
from utils import load_data
import jsonlines
from tasks import Description
from tasks import Description2

exp_name = "0331"
# task_type = Description()
task_type = Description2()  
        
async def task(data, sem):
    async with sem:
        prompt1 = task_type.prompt1()
        try:
            prompt = prompt1.replace("[question]", data["question"])
            pred = await api_forward(prompt)
            pred = task_type.parse1(pred)
            return {"qid": data["qid"], "pred": pred, "prompt": prompt}
        except Exception as e:
            print(e)
            return None

async def eval():
    args = HfArgumentParser(TestingArguments).parse_args_into_dataclasses()[0]
    eval_dataset = build_dataset(
        args.dataset_config, args.eval_dataset, is_training=False
    )
    output_path = f"./outputs/{exp_name}/nextqa.jsonl"
    processed = {}
    sem = asyncio.Semaphore(200)
    if Path(output_path).exists(): 
        processed = load_data(output_path)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()
    invalid = 0
    with jsonlines.open(output_path, "a") as writer:   
        for dataset in eval_dataset.values():
            tasks = [task(data, sem) for data in dataset if data["qid"] not in processed]
            for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                result = await result
                if result is not None:
                    writer.write(result)
                else:
                    invalid += 1
        print(f"Invalid: {invalid}/{len(dataset)}")     

if __name__ == "__main__":
    asyncio.run(eval())
