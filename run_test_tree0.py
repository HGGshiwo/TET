
from runner import async_run_task
import decord
decord.bridge.set_bridge("torch")
from task_utils import api_forward
import asyncio

async def task(**data):
    try:
        out = await api_forward("describe the frame", data["frame"])
    except Exception as e:
        print(e)
        out = None
    if out is None:
        return None
    return {"vid": data["vid"], "idx": data["idx"], "caption": out}

if __name__ == "__main__":
    output_path = "./outputs/0329/nextmc_gpt_4o.jsonl"
    asyncio.run(async_run_task(task, output_path, iter_key="vid", iter_frame=True, video_fps=1))
