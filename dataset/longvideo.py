from .builder import BaseDataset
from utils import *
import json

OPTIONS = ["A", "B", "C", "D", "E", "F"]


class LongVideoDataset(BaseDataset):
    def __init__(self, config, split="test"):
        super().__init__(config, split)

    def get_anno(self):
        assert self.split in ["test", "validation"]
        return load_data(os.path.join(self.config.anno_path, f"{self.split}-00000-of-00001.parquet"))
        
    def build(self):
        data = []
        for item in self.anno:
            qid = f'{item["id"]}_{item["video_id"]}'
            question = item["question"]
            question = [question.strip()]
            for i in range(6):
                question.append(f"{OPTIONS[i]}. {item[f'option{i}'].strip()}")
            question = "\n".join(question)
            new_item = {
                "qid": qid,
                "vid": item['video_id'],
                "video_path": item['video_path'],
                "question": question,
            }
            if item.get("correct_choice", None) is not None:
                new_item["truth"] = OPTIONS[correct_choice]
            data.append(new_item)
        return data
