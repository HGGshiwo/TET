from .builder import BaseDataset
from utils import *
import json

OPTIONS = ["A", "B", "C", "D", "E", "F"]


class MLVUDataset(BaseDataset):
    def __init__(self, config, split="test"):
        super().__init__(config, split)

    def get_anno(self):
        assert self.split == "test"
        with open(self.config.anno_path, "r") as f:
            return json.load(f)
        
    def build(self):
        data = []
        for item in self.anno:
            qid = f'{item["question_id"]}_{item["video"]}'
            question = item["question"]
            question = [question.strip()]
            for i in range(6):
                question.append(f"{OPTIONS[i]}. {str(item['candidates'][i]).strip()}")
            question = "\n".join(question)
            new_item = {
                "qid": qid,
                "vid": item['video'].replace(".mp4", ""),
                "video_path": item['video'],
                "question": question,
            }
            if item.get("answer", None) is not None:
                idx = item["candidates"].index(item["answer"])
                assert idx >= 0 
                new_item["truth"] = OPTIONS[idx]
            data.append(new_item)
        return data
