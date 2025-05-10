from .builder import BaseDataset
from utils import *
from pathlib import Path
import os
from dotenv import load_dotenv
import requests
import time
import ast
from tqdm import tqdm

load_dotenv()
API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
NUM_SECONDS_TO_SLEEP = 5


def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
        },
    ]

    payload = {
        "model": "gpt-3.5-turbo-0125",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                print(
                    f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}"
                )
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            print(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")

        if (
            "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."
            in json.loads(response.content)["error"]["message"]
        ):
            print(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            print(f"All {retries} attempts failed.")
            return "", ""

    return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", 0)
        return int(score)
    except SyntaxError as e:
        print(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0
    except ValueError as e:
        print(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0
    except Exception as e:
        print(
            f"Unexpected error parsing the review string: {e}. Review content: {review}"
        )
        return 0


def parse_acc(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        pred = review_dict.get("pred", "no")
        return str(pred)
    except SyntaxError as e:
        print(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return "no"
    except ValueError as e:
        print(f"Value error parsing the review string: {e}. Review content: {review}")
        return "no"
    except Exception as e:
        print(
            f"Unexpected error parsing the review string: {e}. Review content: {review}"
        )
        return "no"


async def gpt_eval(semaphor, pbar, data_dict):
    evaluated_results = []
    async with semaphor:
        try:
            question = data_dict["question"]
            answer = data_dict["answer"]
            pred = data_dict["pred"]

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval_generic(question, answer, pred, 64)
            score = parse_score(review)
            acc = parse_acc(review)
        except Exception as e:
            print(f"Error: {e}")
            review = "Failed to Get a Proper Review."
            model_name = ""
            score = 0
            acc = "no"

        # Update the dictionary with the new entries
        updated_dict = {
            # "video_name": data_dict["video_name"],
            # "review": review,
            "score": score,
            "acc": acc,
        }
        print(question, answer, pred, score, acc, review)
        pbar.update(1)
        return updated_dict


def moviechat_aggregate(results):
    score, acc = 0, 0
    for result in results:
        eval_score = result["score"]
        try:
            eval_score = int(eval_score)
        except:
            eval_score = 0.0
        score += eval_score

        eval_acc = result["acc"]
        try:
            eval_acc = str(eval_acc)
            if eval_acc == "yes":
                acc += 1
        except:
            acc += 0
    return {"score": score / len(results), "acc": acc / len(results)}


class MovieChatDataset(BaseDataset):
    prompt = "Answer a question using a short phrase or sentence: [Question]"
    
    def get_anno(self):
        split_path_map = {"train": "jsons", "test": "gt/gt"}
        anno_path = Path(self.config.anno_path).joinpath(
            self.split, split_path_map[self.split]
        )
        anno = {}
        for path in anno_path.glob("*.json"):
            try:
                anno[path.stem] = load_data(path)
            except Exception as e:
                import json

                bad_json_str = path.read_text(encoding="utf-8")
                bad_json_str = bad_json_str.split('"caption":')
                fix_json_str = (
                    bad_json_str[0]
                    + '"global":'
                    + bad_json_str[1].split('"global":')[1]
                )
                try:
                    anno[path.stem] = json.loads(fix_json_str)
                except Exception as e:
                    print(f"load {path} error: {e}")
        return anno

    def build(self):
        data = []
        video_path_map = {"train": "raw_videos", "test": "videos"}
        for vid, row in self.anno.items():
            for type in ["breakpoint", "global"]:
                for i, qitem in enumerate(row[type]):
                    video_path = Path(self.split).joinpath(
                        video_path_map[self.split], row["info"]["video_path"]
                    )
                    new_item = {
                        "qid": f"{vid}_{type}_{i}",
                        "vid": vid,
                        "question": self.prompt.replace("[Question]", qitem["question"]),
                        "truth": qitem["answer"],
                        "video_path": str(video_path),
                        "cm_question": qitem["question"], 
                    }
                    if type == "breakpoint":
                        time = int(
                            (qitem["time"] / row["info"]["fps"]) * self.config.frame_fps
                        )
                        time = max(5, time)
                        new_item["time"] = time

                    data.append(new_item)
        return data

    def get_compute_metrics(self, tokenizer):
        import asyncio
        result = []

        def compute_metrics(eval_preds, compute_result):
            inputs = eval_preds.inputs
            preds = tokenizer.batch_decode(
                eval_preds.predictions, skip_special_tokens=True
            )
            for question, pred, truth in zip(
                inputs["cm_question"], preds, inputs["truth"]
            ):
                new_item = dict(question=question, pred=pred, answer=truth)
                result.append(new_item)
            if compute_result:
                with tqdm(total=len(result), desc="gpt jurging") as pbar:
                    loop = asyncio.get_event_loop()
                    semaphore = asyncio.Semaphore(1000)
                    tasks = [gpt_eval(semaphore, pbar, item) for item in result]
                    _result = loop.run_until_complete(asyncio.gather(*tasks))
                    return moviechat_aggregate(_result)
        
        return compute_metrics