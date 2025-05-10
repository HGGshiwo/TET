from .builder import BaseDataset
from utils import *

class EgoSchemaDataset(BaseDataset):
    prompt = "Given the video clip and the question below, choose the most appropriate answer from the five options (A, B, C, D, E). The answer should be based on the context and content of the video. Provide your choice as a letter (A, B, C, D, or E) with no extra outputs. \n\nQuestion: [Question] \n\nOptions:\n A. [OptionA]\n B. [OptionB]\n C. [OptionC]\n D. [OptionD]\n E. [OptionE]"

    def __init__(self, config, split="subset"):
        super().__init__(config, split)

    def get_anno(self):
        anno = load_data(
            self.config.anno_path
        )  # qid --> {question, option 0, option 1, option 2, option 3, option 4, truth (optional)}
        return anno

    def build(self):
        if self.split == "subset":
            json_data = load_data(self.config.subset_path)
            subset_names_list = list(json_data.keys())

        data = []
        for qid, item in self.anno.items():
            if self.split == "subset" and qid not in subset_names_list:
                continue
            question = item["question"]
            choices = [
                item["option 0"],
                item["option 1"],
                item["option 2"],
                item["option 3"],
                item["option 4"],
            ]
            truth = item["truth"] if "truth" in item else -1
            question = self.prompt.replace("[Question]", question)
            question = question.replace("[OptionA]", choices[0])
            question = question.replace("[OptionB]", choices[1])
            question = question.replace("[OptionC]", choices[2])
            question = question.replace("[OptionD]", choices[3])
            question = question.replace("[OptionE]", choices[4])
            new_item = {
                "qid": qid,
                "vid": qid,
                "video_path": qid + ".mp4",
                "question": question,
                "truth": truth,
            }
            data.append(new_item)
        return data