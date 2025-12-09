from .builder import BaseDataset
from utils import *

OPTIONS = ["A", "B", "C", "D", "E"]
class EgoSchemaDataset(BaseDataset):
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
        elif self.split != "full":
            raise ValueError(f"Unknow split: {self.split}, must in [subset, full]")
        data = []
        for item in self.anno:
            qid = item["q_uid"]
            if self.split == "subset" and qid not in subset_names_list:
                continue
            question = item["question"]
            question = [question.strip()]
            for i in range(5):
                question.append(f"{OPTIONS[i]}. {item[f'option {i}'].strip()}")
            question = "\n".join(question)
            new_item = {
                "qid": qid,
                "vid": qid,
                "video_path": qid + ".mp4",
                "question": question,
            }
            if self.split == "subset":
                new_item["truth"] = OPTIONS[json_data[qid]]
            data.append(new_item)
        return data
    
    
    
    