from pathlib import Path
from runner import AsyncRunner
import json
from task_utils import create_model, parse_json
import asyncio 


example = {
    "answer": "A",
    "explain": "put your explaination here",
    "confidence": 3,
}

PROMPT = f"This is a video-related question: [question]. In the absence of a video, choose the answer you think is most correct based on your guess.Output a json format string containing 3 keys: 'answer' and 'explain', 'confidence', where the value corresponding to 'answer' is a single letter (A, B, C, D, E), indicating the answer you choose, the value corresponding to 'explain' is used to explain how you eliminated the wrong options and choose the final answer, and 'confidence' is used to indicate your confidence in the answer, choose from 1, 2, 3. 1 means uncertain, 2 means partially certain, and 3 means very certain. Output example: {json.dumps(example)}"

async def detect_video(runner, **data):
    question = PROMPT.replace("[question]", data["question"])    
    try:
        out = await model.forward(question)
        out = parse_json(out)
        out["qid"] = data["qid"]
        out["prompt"] = question
        out["truth"] = data["truth"]
    except Exception as e:
        print(e)
        out = None
    return out
    

if __name__ == "__main__":
    # exp_name = "0510"
    exp_name = "0522"
    
    dataset_name = "egoschema_subset"
    # dataset_name = "nextmc_test"
    
    model_name = "gpt-4o"
    model = create_model('api', model_name)
    
    output_path = f"./outputs/{exp_name}/filter_{model_name}_{dataset_name}.jsonl"
    runner = AsyncRunner(detect_video, output_path, iter_key="qid", dataset=dataset_name)
    asyncio.run(runner())