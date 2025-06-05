
from runner import AsyncRunner
import decord
decord.bridge.set_bridge("torch")
from task_utils import create_model
import asyncio

async def task(runner, **data):
    try:
        out = await model.forward("describe the frame", data["frame"])
    except Exception as e:
        print(e)
        out = None
    if out is None:
        return None
    return {"vid": data["vid"], "idx": data["idx"], "caption": out}

if __name__ == "__main__":
    # exp_name = "0329"
    exp_name = "0522"
    
    # dataset_name = "nextmc_test"
    dataset_name = "egoschema_subset"
    
    model_name = "gpt-4o"
    model = create_model('api', model_name)
    output_path = f"./outputs/{exp_name}/{dataset_name}_{model_name}.jsonl"
    runner = AsyncRunner(task, output_path, iter_key="vid", iter_frame=True, video_fps=1, dataset=dataset_name)
    asyncio.run(runner())
