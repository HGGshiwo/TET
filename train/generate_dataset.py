import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from runner import AsyncRunner

import asyncio

from utils import create_model, get_cfg
from utils import parse_json
from utils import load_data, save_data, print_cfg
import os


PROMPT = """
**Updated Instruction for COT Data Generation:**

You are a data processing expert. Your task is to generate final Chain-of-Thought (COT) data based on the given input. The input consists of the following fields:
- `raw question`: The original multiple-choice question.
- `key object`: Key objects present in the video.
- `frame select explain`: A detailed explanation of why specific video frames were selected.
- `answer explain`: A step-by-step textual explanation for arriving at the correct answer.
- `keyframe`: A list of key frame indices.
- `answer`: The correct answer letter.

Your output must be a JSON object containing exactly three keys: `"reasoning"`, `"keyframes"`, and `"answer"`.

**Requirements:**
1. **Content & Structure:**
   - **`reasoning`:** An array of **3-6 concise reasoning steps**. Each step should be a single, coherent statement that forms part of a logical, continuous narrative.
   - **`keyframes`:** An array of **3-8 selected key frame indices** (integers) from the input `keyframe` list. You must choose frames that provide the most critical visual evidence to answer the `raw question`. These should:
     - Cover distinct, essential stages or concepts of the answer.
     - Provide maximum information value with minimal redundancy.
     - Be representative of key actions, states, relationships, or transitions mentioned in the `reasoning`.
     - Do not simply output the first or last frames; choose frames that are substantively informative.
   - **`answer`:** The correct answer letter (matching the input).

2. **Reasoning Guidelines:**
   - **First Step(s):** Start with direct observations that reference or are supported by the selected `keyframes`. Mention a few representative frame numbers to ground the reasoning.
   - **Middle Step(s):** Build inferences by connecting observations to the question's core concepts (e.g., purpose, causality, relationships).
   - **Final Step(s):** Explicitly evaluate and eliminate incorrect options, leading to the final answer. Option evaluation should be concise and integrated into the narrative.
   - **General:** Remove redundant details, meta-commentary (e.g., "I observe that"), and process descriptions. Be direct and factual.

3. **Core Principle:** The `reasoning` must be **self-contained** and **fully answer the `raw question`** using the logic and evidence synthesized from the inputs. The selected `keyframes` should visually substantiate the `reasoning`.

**Output Example:**

```json
{{
  "reasoning": [
    "Key frames (e.g., 7, 19) show C sieving grain, which cleans or sorts it.",
    "Subsequent frames (116, 124) show her washing the grain in water, a typical preparatory step for cooking.",
    "Finally, frame 178 shows the grain placed near a stove with a matchbox, indicating the cooking stage.",
    "The entire sequence—sieving, washing, moving to stove—demonstrates that the objective is preparation for cooking, making D correct, while A and B are incomplete steps, C is a temporary state, and E is unsupported."
  ],
  "keyframes": [7, 19, 116, 124, 178],
  "answer": "D"
}}
```
**Input Data:**
`raw question`: {question}
`frame select explain`: {frame_select_explain}
`answer explain`: {answer_explain}
`keyframe`: {keyframe}
`answer`: {answer}
"""


def frame_filter(runner, data):
    qid = data["qid"]
    _d = runner.step5_answer_data[qid]
    return _d["truth"] == _d["answer"]


async def task(runner, **data):
    try:
        qid = data["qid"]
        format_kwargs = {}
        format_kwargs["question"] = data["question"]
        # format_kwargs["key_object"] = runner.step1_obj_data[qid]
        format_kwargs["frame_select_explain"] = [
            "\n".join(
                d["explain"] for d in runner.step4_select2_data[qid]["raw_output"]
            )
        ]
        format_kwargs["answer_explain"] = runner.step5_answer_data[qid]["explain"]
        format_kwargs["keyframe"] = runner.step5_answer_data[qid]["input_idx"]
        format_kwargs["answer"] = runner.step5_answer_data[qid]["truth"]

        _prompt = PROMPT.format(**format_kwargs)
        out = await model.forward(_prompt)
        out = parse_json(out)
    except Exception as e:
        print(e)
        out = None
    if out is not None:
        return {"qid": data["qid"], **out}
    return None


if __name__ == "__main__":

    cfg = load_data(Path(__file__).parent.joinpath("./config/generate_dataset.yml"))
    exp_name = cfg["exp_name"]
    print_cfg(cfg)
    model_name = cfg["model_name"]
    save_data(
        cfg,
        Path(__file__).parent.joinpath(f"./outputs/{exp_name}/generate_dataset.yml"),
    )

    output_dir = Path(__file__).parent.joinpath(f"./outputs/{exp_name}")
    model = create_model("api", model_name)

    data_path = Path(__file__).parent.parent.joinpath("outputs")
    for answer_path in cfg["answer_path"]:
        obj_cfg, dino_cfg, select_cfg, select2_cfg, answer_cfg = get_cfg(answer_path)
        dataset_name = obj_cfg["dataset_name"]
        output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
        runner = AsyncRunner(
            task, output_path, iter_key="qid", dataset=dataset_name, filter=frame_filter
        )
        runner.step1_obj_data = load_data(
            data_path.joinpath(obj_cfg["exp_name"], "obj.jsonl")
        )
        # runner.step3_select_data = load_data(data_path.joinpath(select_cfg['exp_name'], "select.jsonl"))
        runner.step4_select2_data = load_data(
            data_path.joinpath(select2_cfg["exp_name"], "select2.jsonl")
        )
        runner.step5_answer_data = load_data(
            data_path.joinpath(answer_cfg["exp_name"], "answer.jsonl")
        )
        asyncio.run(runner())
