from runner import AsyncRunner

import json
import asyncio
from task_utils import api_forward
from task_utils import parse_json

class_list = open("yolo_class.txt", "r").read()
example2 = {"person": {"min": 2}, "bicycle": {"max": 3, "min": 1}}
prompt = f"Below is a question related to video: [question]. Analyze the objects that may appear in the frames related to this question, as well as their quantity limits. The objects need to be selected from the list below: {class_list}. Finally, output a JSON formatted string as the result, which includes several key-value pairs. The key is the name of the object that may appear, and the value is a dictionary containing two keys: 'max' and 'min', representing the maximum and minimum quantity limits, respectively. If there is no limit, one of these keys can be omitted. If no object can be obtained from the quesiton, output a empty json. The output example is: {json.dumps(example2)}"


async def task(runner, **data):
    try:
        _prompt = prompt.replace("[question]", data["question"].split("A. ")[0])
        out = await api_forward(_prompt)
        out = parse_json(out)
    except Exception as e:
        print(e)
        out = None
    if out is not None:
        return {"qid": data["qid"], "maxmin": out}


if __name__ == "__main__":
    output_path = "./outputs/0411/yolo_maxmin.jsonl"
    runner = AsyncRunner(task, output_path, iter_key="qid")
    asyncio.run(runner())
