from runner import AsyncRunner

import asyncio
from utils import create_model
from utils import parse_json
from utils import load_data, save_data

prompt = "Below is a question related to a video. Please analyze the people or objects that will appear in the scene asked about in the question. Only use the information from the question itself, do not add objects based on imagination. If it is not possible to obtain any people or objects from the question, return an empty list. Output a JSON list containing the names of the objects that appear in the question; Here is the question: [question]"

prompt2 = 'Below is a question related to a video along with its corresponding options. Please analyze the people or objects that would appear in the scenarios mentioned in both the question and the options. Only use known information; do not imagine or add objects that are not provided. Output a JSON string, where the keys are "question", "A", "B", "C", "D", and "E", each corresponding to a list of objects appearing in the question and options. If no people or objects can be identified from the known information, return an empty list. The question and options are as follows: [question]'

prompt3 = """
The following is a video-related question along with its options. Please analyze the people or objects mentioned in both the question and the options. For each extracted object, please ensure that:

1. The person or object is indeed mentioned in the text, not imagined.
2. The person or object must be visible in the footage; invisible things such as music are not allowed.
3. The person or object has certain characteristics that allow it to be located in the footage. For example, 'second woman' cannot be directly located in the footage, but 'woman in blue' can.
3. The person or object must have a clear meaning and cannot be overly broad, such as 'something'.
Only add objects that meet these criteria to the list. Finally, output a JSON string, with keys 'question', 'A', 'B', 'C', 'D', and 'E', corresponding to the lists of objects mentioned in the question and each option, respectively. If no people or objects can be identified from the given information, return an empty list. The question and options are as follows:[question]
"""

prompt4 = """
Below is a question related to a video and its options. Please analyze the people or objects mentioned in the question and options. For each extracted entity, please ensure that:

1. The person or object is explicitly mentioned in the text and is not fabricated.
2. The person or object must be visible in the video frame; invisible entities such as music are not allowed.
3. Pronouns should be converted to the name of the object, for example, "he" should be changed to "man".
4. The person or object must have a clear and specific meaning; overly broad terms such as "something" are not allowed.
5. If the question contains a description of a person, convert it into a concise adjective + noun phrase.
Only add objects that meet these criteria to the list. Finally, output a JSON string with the keys "question", "A", "B", "C", "D", and "E", corresponding to the list of objects mentioned in the question and each option. If no people or objects can be identified based on the given information, return an empty list. The question and options are as follows: [question]
"""


async def task(runner, **data):
    try:
        if option_type is None:
            quesiton, option = data["question"].split("A. ")
            option = "A. " + option
        else:
            quesiton = data["question"]

        if option_type == "option2":
            _prompt = prompt4
        elif option_type == "option1":
            _prompt = prompt2
        else:
            _prompt = prompt
        _prompt = _prompt.replace("[question]", quesiton)
        _prompt = f"NOTE: you should think step by step before give the final answer.\n{_prompt}"
        out = await model.forward(_prompt)
        out = parse_json(out)
    except Exception as e:
        print(e)
        out = None
    if out is not None:
        return {"qid": data["qid"], "pred": out}
    return None


if __name__ == "__main__":
    cfg = load_data("./configs/obj.yml")
    exp_name = cfg["exp_name"]
    dataset_name = cfg["dataset_name"]
    option_type = cfg["option_type"]
    model_name = cfg["model_name"]
    save_data(cfg, f"./outputs/{exp_name}/obj.yml")

    output_path = f"./outputs/{exp_name}/obj.jsonl"
    model = create_model("api", model_name)
    runner = AsyncRunner(task, output_path, iter_key="qid", dataset=dataset_name)
    asyncio.run(runner())
