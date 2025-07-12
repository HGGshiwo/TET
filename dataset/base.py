from torch.utils.data import Dataset
import argparse
from transformers import EvalPrediction
from pathlib import Path
from utils import load_data

class BaseDataset(Dataset):
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", cls.__name__.lower().replace("dataset", ""))
        BaseDataset.registry[name] = cls

    @classmethod
    def create(cls, config, key):
        key = key.lower().split("_")
        name = key[0]
        split = key[1] if len(key) > 1 else None
        subclass = cls.registry.get(name)
        if subclass:
            return subclass(config[name], split=split)
        else:
            raise ValueError(f"Key Must in {list(cls.registry.keys())}, got {name}")

    def __init__(self, config, split):
        """
        num_examples_to_run < 0: run all
        """
        config = argparse.Namespace(**config)
        self.config = config
        self.split = split
        self.anno = self.get_anno()

        data = self.build()
        data = self.filter(data, config.num_examples_to_run)
        self.v2q_map = {}
        for item in data:
            qid, vid = item["qid"], item["vid"]
            if vid not in self.v2q_map:
                self.v2q_map[vid] = []
            self.v2q_map[vid].append(qid)

        self.data = data

    def get_descriptions(self):
        raise NotImplementedError

    # def format_narration(self, narr):
    #     raise NotImplementedError

    def filter(self, data, num_examples_to_run):
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
        return data
    
    def get_question(self, vid):
        return self.v2q_map[vid]

    def get_video_info(self):
        video_info = list(
            {
                d["vid"]: dict(vid=d["vid"], video_path=d["video_path"], qid=self.get_question(d["vid"]))
                for d in self.data
            }.values()
        )
        return video_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_compute_metrics(self, tokenizer):
        return NotImplementedError