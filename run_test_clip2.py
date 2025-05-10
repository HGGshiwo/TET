# 2. 给出起始帧描述，然后让模型给出问题
from runner import Runner
from pathlib import Path
from task_utils import create_model, get_frame
from utils import load_data
import torch

def frame_select(runner, **data):
    qid = data["qid"]
    video_path = Path(runner.dataset.config.video_path).joinpath(data["video_path"])
    video = get_frame(video_path, 1)
    obj = select_data[qid]["obj"]
    sim = model.forward(obj, video)
    sim = sim.max(dim=-1).values
    sim = torch.topk(sim, int(0.5 * sim.shape[0]), dim=0).indices
    sim = [int(i) for i in sim.cpu().numpy()]
    sim = sorted(sim)
    return {"qid": qid, "relevant_idx": sim, "last": len(video)}


if __name__ == "__main__":
    output_path = "./outputs/0508/clip.jsonl"
    data_path = "./outputs/0508/object.jsonl"
    select_data = load_data(data_path)
    model = create_model("clip")
    
    runner = Runner(frame_select, output_path, iter_key="qid")
    runner()
