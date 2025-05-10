from dataset.builder import build_dataset
from args import TestingArguments
from transformers import HfArgumentParser
from pathlib import Path
import decord

decord.bridge.set_bridge("torch")
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from utils import load_data
from task_utils import api_forward, list2dict
import asyncio
import jsonlines
import re
import json

load_dotenv()
client = OpenAI()


output = {
    "green cup": ["green cup is in baby's hand", "green cup is on the floor"],
    "baby": ["baby hold the green cup", "baby clap proudly", "baby lay on floor", "baby picked the cup up", "baby crawl"],
    "lady": ["lady sitting down"],
}
output2 = {
    "green cup": ["green cup is in baby's hand"],
    "baby": ["baby hold the green cup", "baby is drinking"],
    "lady": ["disappear"]
}
prompt1 = f"There is a video-related question and option. Please splits them into descriptions of someone doing something, and finally outputs a Json string, extracting the same object as the key and the corresponding description list of the object as the value. Use full names of objects rather than pronouns in descriptions.Inputs:\nwhat did the baby do after throwing the green cup away while on the floor near the end?\nA.clap proudly\nB.the lady sitting down\nC.lay on floor\nD.	just picked it up\nE.crawl\nOutputs:\n{json.dumps(output)}\nInputs:\n[question]\nOutputs:\n"

prompt2 = f"This is a frame in a complete video. Below is a string in Json format. The key represents an object, and the value represents the state list of the object. First, determine whether the object corresponding to the key appears in the picture. If not, the state of the object is \"disappear\". If it appears, select several states that match the picture in the state list. If there is no matching state, output the description of the object state, and finally output a json file. The key represents the object, and the value represents the current state of the object.Inputs:\n{output}\nOutputs:\n{output2}\nInputs:\n[question]\nOutputs:\n"

def parse_json(pred):
    pred = pred.replace("```json", "").replace("```", "")
    try:
        pred = json.loads(pred)
    except json.JSONDecodeError:
        try:
            pred = pred.split("{")[1].split("}")[0]
            pred = "{" + pred + "}"
            pred = json.loads(pred)
        except:
            return None
    return pred

async def run_sigle(pred, frame):
    text = prompt2.replace("[question]", json.dumps(pred))
    out = await api_forward(text, frame)
    if out is not None:
        out = parse_json(out)
    return out

async def eval():
    args = HfArgumentParser(TestingArguments).parse_args_into_dataclasses()[0]

    eval_dataset = build_dataset(
        args.dataset_config, args.eval_dataset, is_training=False
    )
    output_path = "./outputs/nextqa_all.jsonl"
    output_path1 = "./outputs/nextqa_all1.jsonl"
    video_path = "D:/datasets/nextqa/NExTVideo"
    with jsonlines.open(output_path, "a") as writer, jsonlines.open(output_path1, "a") as writer1:
        for dataset in eval_dataset.values():
            # input_path = "./outputs/nextqa.jsonl"
            if Path(output_path1).exists():
                processed1 = load_data(output_path1)
            else:
                processed1 = {}
                Path(output_path1).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path1).touch()
            if Path(output_path).exists():
                list_data = jsonlines.open(output_path, "r")
                from collections import defaultdict
                processed = defaultdict(dict)
                for data in list_data:
                    processed[data["qid"]][data["idx"]] = data
            else:
                processed = {}
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).touch()
            # input_data = load_data(input_path)
            # tasks = []
            # sem = asyncio.Semaphore(200)
            for data in dataset:
                # if data["qid"] not in input_data:
                    # continue
                # pred = input_data[data["qid"]]["question"]
                if data["qid"] in processed1:
                    pred = processed1[data["qid"]]
                else:
                    pred = await api_forward(prompt1.replace("[question]", data["question"]))
                    pred = parse_json(pred)
                    pred["qid"] = data["qid"]
                    writer1.write(pred)
                _video_path = Path(video_path).joinpath(data["video_path"])
                # scenes = detect(str(video_path), ContentDetector())
                vr = decord.VideoReader(str(_video_path))

                fps = vr.get_avg_fps()
                # prompt = 'If the object related to the description is not visible, reply that the object is invisible. otherwise answer "True" if the description and the picture match, "False" if they are not, split the answer by line breaks. Example: \nHere are 3 descriptions:\nThe girl is washing a car.\nThe cat is running\nThe girl is running\nOutput:\nTrue\nthe cat is invisible\nFalse\nHere are [len] descriptions:\n[question]'
                sub_task = []
                frame_idx = []
                for idx in range(0, len(vr), int(fps)):
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
                    text = prompt2.replace("[question]", json.dumps(pred))
                    out = await api_forward(text, frame)
                    if out is not None:
                        out = parse_json(out)
                        if out is None:
                            continue
                        writer.write(
                            {
                                "qid": data["qid"],
                                "pred": out,
                                "idx": idx,
                            }
                        )

            # print(f"Invalid: {invalid}/{total}")


if __name__ == "__main__":
    asyncio.run(eval())
