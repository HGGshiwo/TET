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

        self.depth_res = None
        self.width_res = None
        self.durations = None

        if config.width_res_path is not None and Path(config.width_res_path).exists():
            self.width_res = load_data(config.width_res_path)
        if config.depth_res_path is not None and Path(config.depth_res_path).exists():
            self.depth_res = load_data(config.depth_res_path)
        # if config.data_path is not None and Path(config.data_path).exists():
        #     self.narrations = self.get_descriptions()

        data = self.build()
        data = self.filter(data, config.num_examples_to_run)
        self.v2q_map = {}
        for item in data:
            qid, vid = item["qid"], item["vid"]
            if vid not in self.v2q_map:
                self.v2q_map[vid] = []
            self.v2q_map[vid].append(qid)
            if self.width_res is not None and qid in self.width_res:
                item.update(self.width_res[qid])
            if self.depth_res is not None and qid in self.depth_res:
                item.update(self.depth_res[qid])

            item["feature_path"] = config.feature_path
            item["frame_path"] = config.frame_path

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
        result = dict(num_totals=0, num_corrects=0)
        def compute_metrics(eval_pred: EvalPrediction, return_metrics):
            inputs = eval_pred.inputs
            inputs_truth = inputs.pop("truth")
        
            for pred, label in zip(eval_pred.predictions, inputs_truth):
                pred = tokenizer.decode(pred[pred > 0], skip_special_tokens=True)
                result['num_totals'] += 1
                if pred == label:
                    result['num_corrects'] += 1
            if return_metrics:
                stat = {
                    "acc": result['num_corrects'] / result['num_totals'],
                }
                return stat
        return compute_metrics