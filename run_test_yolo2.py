from runner import AsyncRunner

import json
import asyncio
from task_utils import create_model
from task_utils import parse_json

class_list = open("yolo_class.txt", "r").read()
example2 = {"person": {"min": 2}, "bicycle": {"max": 3, "min": 1}}
prompt = f"Below is a question related to video: [question]. Analyze the objects that may appear in the frames related to this question, as well as their quantity limits. The objects need to be selected from the list below: {class_list}. Finally, output a JSON formatted string as the result, which includes several key-value pairs. The key is the name of the object that may appear, and the value is a dictionary containing two keys: 'max' and 'min', representing the maximum and minimum quantity limits, respectively. If there is no limit, one of these keys can be omitted. If no object can be obtained from the quesiton, output a empty json. The output example is: {json.dumps(example2)}"


async def task(runner, **data):
    try:
        _prompt = prompt.replace("[question]", data["question"].split("A. ")[0])
        out = await model.forward(_prompt)
        out = parse_json(out)
    except Exception as e:
        print(e)
        out = None
    if out is not None:
        return {"qid": data["qid"], "maxmin": out}
    return None


if __name__ == "__main__":
    # exp_name = "0411"
    exp_name = "0522"
    # dataset_name = "nextmc_test"
    dataset_name = "egoschema_subset"
    
    model_name = "gpt-4o"
    
    output_path = f"./outputs/{exp_name}/yolo_maxmin_{model_name}.jsonl"
    model = create_model('api', model_name)
    runner = AsyncRunner(task, output_path, iter_key="qid", dataset=dataset_name)
    asyncio.run(runner())
