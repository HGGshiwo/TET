import decord
from PIL import Image
from pathlib import Path
import json
decord.bridge.set_bridge("torch")
from runner import Runner

def run_task(runner, **data):
    qid = data["qid"]
    return {"qid": qid, "vid": data["vid"], "relevant": [], "question": data["question"], "truth": data["truth"]}

if __name__ == "__main__":
    runner = Runner(run_task, iter_key="qid", output_path="outputs/relevant.jsonl")
    runner()