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
    
    def get_compute_metrics2(self):
        results = dict(total_num=0, correct_num=0)
        failed = []
        def compute_metrics(pred, item, compute_result):
            parsed_pred = self.parse_multi_choice_response(pred)
            if parsed_pred is None:
                print("None", item["qid"])
            if parsed_pred == item["truth"]:
                results["correct_num"] += 1
            else:
                # print(item["qid"])
                failed.append(item["qid"])
            results["total_num"] += 1
            
            if compute_result:
                return {
                    "acc": results["correct_num"] / results["total_num"],
                    "failed": failed,
                }

        return compute_metrics
    
    def parse_multi_choice_response(self, response):
        """
        Parse the prediction from the generated response.
        Return the predicted index e.g., A, B, C, D.
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
        """
        candidates = []
        all_choices = ["A", "B", "C", "D", "E"]
        response = response.replace("*", "")
        str_list = [r.strip() for r in response.split("\n") if r.strip() != ""]
        for res in str_list:
            res = res.split(":")[-1].strip()
            if res in all_choices:
                candidates.append(res)
                break
        
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        
        response = " " + response + " "  # add space to avoid partial match

        index_ans = True
        ans_with_brack = False
        
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True

        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A B C D
                if f"{choice} " in response:
                    candidates.append(choice)

        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A. B. C. D.
                if f"{choice}." in response:
                    candidates.append(choice)

        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A. B. C. D.
                if f"{choice}:" in response:
                    candidates.append(choice)
        
        if len(candidates) == 0:
            for choice in all_choices:  # e.g., A. B. C. D.
                if response.strip().startswith(choice):
                    candidates.append(choice)
                                
        # # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
        # if len(candidates) == 0 and len(response.split()) > 5:
        #     for index, ans in index2ans.items():
        #         if ans.lower() in response.lower():
        #             candidates.append(index)
        #             index_ans = False  # it's content ans.

        if len(candidates) == 0:  # still not get answer, randomly choose one.
            # pred_index = random.choice(all_choices)
            # pred_index = all_choices[0]  # use the first one as default
            pred_index = None
        elif len(candidates) > 1:
            pred_index = None
        #     start_indexes = []
        #     if index_ans:
        #         if ans_with_brack:
        #             for can in candidates:
        #                 index = response.rfind(f"({can})")
        #                 start_indexes.append(index)  # -1 will be ignored anyway
        #             # start_indexes = [generated_response.index(f'({can})') for can in candidates]
        #         else:
        #             for can in candidates:
        #                 index = response.rfind(f" {can} ")
        #                 start_indexes.append(index)
        #     else:
        #         for can in candidates:
        #             index = response.lower().rfind(index2ans[can].lower())
        #             start_indexes.append(index)
        #     # get the last one
        #     pred_index = candidates[np.argmax(start_indexes)]
        else:  # if only one candidate, use it.
            pred_index = candidates[0]

        return pred_index