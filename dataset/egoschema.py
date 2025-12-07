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