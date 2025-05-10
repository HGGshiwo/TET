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


load_dotenv()
client = OpenAI()


async def run_task(data, prompt, sem):
    async with sem:
        try:
            out = await api_forward(prompt)
            answer = out.replace("*", "")
            answer = re.sub(r"Answers:", "answer:", out, flags=re.IGNORECASE)
            answer = re.split(r"answer:", answer, flags=re.IGNORECASE)[1]
            answer = answer.replace("\n", "").strip()
        except Exception as e:
            out, answer = None, None
            print(out, e)
        return data, {
            "qid": data["qid"],
            "pred": out,
            "prompt": prompt,
            "answer": answer,
            "truth": data["truth"],
        }


async def eval():
    args = HfArgumentParser(TestingArguments).parse_args_into_dataclasses()[0]

    eval_dataset = build_dataset(
        args.dataset_config, args.eval_dataset, is_training=False
    )
    output_path = "./outputs/nextqa3.jsonl"
    with jsonlines.open(output_path, "a") as writer:
        for dataset in eval_dataset.values():
            input_path = "./outputs/nextqa2.jsonl"
            input_path1 = "./outputs/nextqa.jsonl"
            if Path(output_path).exists():
                processed = load_data(output_path)
            else:
                processed = {}
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).touch()
            input_data = list2dict(input_path)
            input_quesiton = load_data(input_path1)
            compute_metrics = dataset.get_compute_metrics2()
            tasks = []
            sem = asyncio.Semaphore(200)
            for data in dataset:
                if data["qid"] not in input_data:
                    continue
                if data["qid"] in processed:
                    out = processed[data["qid"]]["pred"].split("\n")[-1]
                    metrics = compute_metrics(out, data, True)
                    continue

                ans = input_data[data["qid"]]
                que = input_quesiton[data["qid"]]["question"]
                prompt = "There is a question related to a [time] seconds video: [question]. The following text describes the change of the object state every 1 second in the video. The format is frame number: object state description. If the state of the object has not changed, the frame number at that time point is omitted.\n[frames].\nThe timing of the question may fall within an event description or between event descriptions. If it occurs between event descriptions, infer the answer based on the preceding and following descriptions. Analyze between which descriptions the following question occurs, and answer the question based on the corresponding descriptions. Provide the analysis and the final answer, separated by a line break. Output example: Analysis: analysis\nAnswer: answer"
                frames, last_state = [], {}
                for i, idx_a in enumerate(ans):
                    idx, a = idx_a
                    sub_frames = []
                    for sub_ans, sub_que in zip(a, que):
                        if re.search("false", sub_ans, flags=re.IGNORECASE) is not None:
                            continue
                        last_ans = last_state.get(sub_que, None)
                        last_invisble = last_ans is not None and "invisible" in last_ans.lower()
                        cur_invisible = "invisible" in sub_ans.lower()
                        if last_ans != sub_ans and not (last_invisble and cur_invisible):
                            sub_frames.append(f"{sub_que}")
                        last_state[sub_que] = sub_ans
                    if len(sub_frames) != 0:
                        frames.append((i, ",".join(sub_frames)))
                frames = "\n".join(
                    [f"{idx_frame[0]}: {idx_frame[1]}" for idx_frame in frames]
                )
                _prompt = prompt.replace("[frames]", frames)
                _prompt = _prompt.replace("[question]", data["question"])
                _prompt = _prompt.replace("[time]", str(len(ans)))
                tasks.append(run_task(data, _prompt, sem))
            total, invalid = len(tasks), 0
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                data, out = await task
                if out["pred"] is not None:
                    writer.write(out)
                    out = out["pred"].split("\n")[-1]
                    metrics = compute_metrics(out, data, True)
                else:
                    invalid += 1
            print(f"Invalid: {invalid}/{total}")
            print(metrics)


if __name__ == "__main__":
    asyncio.run(eval())
