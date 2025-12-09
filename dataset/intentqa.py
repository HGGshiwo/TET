from .builder import BaseDataset
from utils import *
import pandas as pd

OPTIONS = ["A", "B", "C", "D", "E"]


class IntentQADataset(BaseDataset):
    def __init__(self, config, split="test"):
        super().__init__(config, split)

    def get_anno(self):
        anno_path = os.path.join(self.config.anno_path, f"{self.split}.csv")
        df = pd.read_csv(anno_path)
        anno = []
        for row in df.itertuples():
            data = {
                "vid": row.video_id,
                "qid": f"{row.video_id}_{row.qid}",
                "question": row.question,
                "options": [getattr(row, f"a{i}") for i in range(5)]
            }
            if hasattr(row, "answer"):
                data["truth"] = row.answer
            anno.append(data)
        return anno

    def build(self):
        data = []
        for item in self.anno:
            qid = item["qid"]
            question = item["question"]
            question = [question.strip()]
            for i in range(5):
                question.append(f"{OPTIONS[i]}. {item['options'][i].strip()}")
            question = "\n".join(question)
            new_item = {
                "qid": qid,
                "vid": item['vid'],
                "video_path": f"{item['vid']}.mp4",
                "question": question,
            }
            if item.get("truth", None) is not None:
                new_item["truth"] = OPTIONS[item["truth"]]
            data.append(new_item)
        return data
