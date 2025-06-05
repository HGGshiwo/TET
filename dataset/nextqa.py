from .builder import BaseDataset
import pandas as pd
from utils import *
import random
from transformers import EvalPrediction


class NextQADataset(BaseDataset):
    def __init__(self, config, split="test"):
        super().__init__(config, split)

    def get_anno(self):
        # MC: video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
        return pd.read_csv(f"{self.config.anno_path}/{self.split}.csv")


OPTIONS = ["A", "B", "C", "D", "E"]


class NextMCDataset(NextQADataset):

    def parse_multi_choice_response(self, response, all_choices, index2ans):
        """
        Parse the prediction from the generated response.
        Return the predicted index e.g., A, B, C, D.
        https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
        """
        candidates = []
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
                                
        # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
        if len(candidates) == 0 and len(response.split()) > 5:
            for index, ans in index2ans.items():
                if ans.lower() in response.lower():
                    candidates.append(index)
                    index_ans = False  # it's content ans.

        if len(candidates) == 0:  # still not get answer, randomly choose one.
            # pred_index = random.choice(all_choices)
            # pred_index = all_choices[0]  # use the first one as default
            pred_index = None
        elif len(candidates) > 1:
            start_indexes = []
            if index_ans:
                if ans_with_brack:
                    for can in candidates:
                        index = response.rfind(f"({can})")
                        start_indexes.append(index)  # -1 will be ignored anyway
                    # start_indexes = [generated_response.index(f'({can})') for can in candidates]
                else:
                    for can in candidates:
                        index = response.rfind(f" {can} ")
                        start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.lower().rfind(index2ans[can].lower())
                    start_indexes.append(index)
            # get the last one
            pred_index = candidates[np.argmax(start_indexes)]
        else:  # if only one candidate, use it.
            pred_index = candidates[0]

        return pred_index

    def build(self):

        vid_map = load_data(self.config.map_path)
        data = []
        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            vid = str(row["video"])
            question, truth = row["question"], row["answer"]
            _qid, q_type = row["qid"], row["type"]
            qid = f"{vid}_{_qid}"
            question = [question.strip()]
            for i in range(5):
                question.append(f"{OPTIONS[i]}. {row[f'a{i}'].strip()}")
            question = "\n".join(question)
            item = {
                "qid": qid,
                "vid": vid,
                "video_path": vid_map[vid] + ".mp4",
                "q_type": q_type,
                "question": question,
                "truth": OPTIONS[truth],
            }
            for i in range(5):
                item[f"cm_a{i}"] = row[f"a{i}"]  # for computing metrics
            data.append(item)
        return data

    def get_multi_choice_info(self, doc):
        all_choices = []
        index2ans = {}
        for i in range(5):
            index2ans[OPTIONS[i]] = doc[f"a{i}"].strip()
            all_choices.append(OPTIONS[i])

        return index2ans, all_choices
    
    
    def get_compute_metrics(self, tokenizer):
        results = dict(total_num=0, correct_num=0)
        score_acc = []

        def compute_metrics(eval_results: EvalPrediction, compute_result):
            inputs = eval_results.inputs
            inputs_truth = inputs.pop("truth")

            predictions = eval_results.predictions
            rel_accs = eval_results.losses
            for i, (pred, answer, acc) in enumerate(
                zip(predictions, inputs_truth, rel_accs)
            ):
                pred = tokenizer.decode(pred, skip_special_tokens=True)
                doc = {f"a{j}": inputs[f"cm_a{j}"][i] for j in range(5)}
                index2ans, all_choices = self.get_multi_choice_info(doc)
                parsed_pred = self.parse_multi_choice_response(
                    pred, all_choices, index2ans
                )

                if parsed_pred == answer:
                    results["correct_num"] += 1
                results["total_num"] += 1
                score_acc.append(acc)

            if compute_result:
                return {
                    "acc": results["correct_num"] / results["total_num"],
                    "score_acc": sum(score_acc) / len(score_acc),
                }

        return compute_metrics

    def get_compute_metrics2(self):
        results = dict(total_num=0, correct_num=0)
        failed_list = []
        def compute_metrics(pred, item, compute_result):
            doc = {f"a{j}": item[f"cm_a{j}"] for j in range(5)}
            index2ans, all_choices = self.get_multi_choice_info(doc)
            parsed_pred = self.parse_multi_choice_response(
                pred, all_choices, index2ans
            )
            if parsed_pred is None:
                print(f"{item['qid']} is None")
            if parsed_pred == item["truth"]:
                results["correct_num"] += 1
            else:
                failed_list.append(item["qid"])
            results["total_num"] += 1
            
            if compute_result:
                return {
                    "acc": results["correct_num"] / results["total_num"],
                    "failed": failed_list,
                }

        return compute_metrics

class NextOEDataset(NextQADataset):
    prompt = "Answer a question using a short phrase or sentence: [Question]"

    def build(self):
        map = load_data(self.config.map_path)
        data = []
        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            vid = str(row["video"])
            question, truth = row["question"], row["answer"]
            _qid, q_type = row["qid"], row["type"]
            qid = f"{vid}_{_qid}"
            data.append(
                {
                    "qid": qid,
                    "vid": vid,
                    "video_path": map[vid] + ".mp4",
                    "q_type": q_type,
                    "question": self.prompt.replace("[Question]", question),
                    "truth": truth,
                }
            )

        return data

    def get_compute_metrics(self, tokenizer):

        stopwords = set(pd.read_csv(self.config.stopwords_path).squeeze())
        try:
            import nltk
            from nltk.corpus import wordnet
            from nltk.tokenize import word_tokenize

            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except ImportError:
            print(
                "nltk not installed. Please install nltk to use this module. You can install it by running 'pip install nltk'"
            )

        try:
            from pywsd.utils import lemmatize_sentence
        except ImportError:
            print(
                "pywsd not installed. Please install pywsd to use this module. You can install it by running 'pip install pywsd'"
            )

        def wup(word1, word2, alpha):
            """
            calculate the wup similarity
            :param word1:
            :param word2:
            :param alpha:
            :return:
            """
            # print(word1, word2)
            if word1 == word2:
                return 1.0

            w1 = wordnet.synsets(word1)
            w1_len = len(w1)
            if w1_len == 0:
                return 0.0
            w2 = wordnet.synsets(word2)
            w2_len = len(w2)
            if w2_len == 0:
                return 0.0

            # match the first
            word_sim = w1[0].wup_similarity(w2[0])
            if word_sim is None:
                word_sim = 0.0

            if word_sim < alpha:
                word_sim = 0.1 * word_sim
            return word_sim

        def wups(words1, words2, alpha):
            """

            :param pred:
            :param truth:
            :param alpha:
            :return:
            """
            sim = 1.0
            flag = False
            for w1 in words1:
                max_sim = 0
                for w2 in words2:
                    word_sim = wup(w1, w2, alpha)
                    if word_sim > max_sim:
                        max_sim = word_sim
                if max_sim == 0:
                    continue
                sim *= max_sim
                flag = True
            if not flag:
                sim = 0.0
            return sim

        def get_wups(pred, truth, alpha):
            """
            calculate the wups score
            :param pred:
            :param truth:
            :return:
            """
            pred = word_tokenize(pred)
            truth = word_tokenize(truth)
            item1 = wups(pred, truth, alpha)
            item2 = wups(truth, pred, alpha)
            value = min(item1, item2)
            return value

        def remove_stop(sentence):
            sentence.replace("</s>", "")  # video-llava
            words = lemmatize_sentence(sentence)
            words = [w for w in words if not w in stopwords]
            return " ".join(words)

        qtypes = ["CW", "CH", "TN", "TC", "DB", "DC", "DL", "DO"]
        num = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
        over_num = {"C": 0, "T": 0, "D": 0}
        wups0 = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
        wups9 = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
        results = dict(ref_num=0)

        def compute_metrics(eval_results, compute_result):
            inputs = eval_results.inputs
            inputs_truth = inputs.pop("cm_truth")
            inputs_qtype = inputs.pop("cm_q_type")
            predictions = eval_results.predictions
            inputs_add_ref_anse = inputs.pop("cm_additional_ref_answer", None)
            for i, (pred, answer, qtype) in enumerate(
                zip(predictions, inputs_truth, inputs_qtype)
            ):
                pred = tokenizer.decode(pred, skip_special_tokens=True)
                pred_ans = remove_stop(pred)
                gt_ans = remove_stop(answer)
                if qtype == "TP":
                    qtype = "TN"

                if inputs_add_ref_anse is not None:
                    add_ref_ans = inputs_add_ref_anse[i]
                    add_ref_ans = remove_stop(add_ref_ans)
                    if qtype == "DC" or qtype == "DB":
                        cur_0 = (
                            1 if pred_ans == gt_ans or pred_ans == add_ref_ans else 0
                        )
                        cur_9 = cur_0
                    else:
                        cur_0 = max(
                            get_wups(pred_ans, gt_ans, 0),
                            get_wups(pred_ans, add_ref_ans, 0),
                        )
                        cur_9 = max(
                            get_wups(pred_ans, gt_ans, 0.9),
                            get_wups(pred_ans, add_ref_ans, 0),
                        )
                else:
                    if qtype == "DC" or qtype == "DB":
                        cur_0 = 1 if pred_ans == gt_ans else 0
                        cur_9 = cur_0
                    else:
                        cur_0 = get_wups(pred_ans, gt_ans, 0)
                        cur_9 = get_wups(pred_ans, gt_ans, 0.9)

                num[qtype] += 1
                over_num[qtype[0]] += 1
                results["ref_num"] += 1
                wups0[qtype] += cur_0
                wups9[qtype] += cur_9

            if compute_result:
                wups0_all = wups9_all = 0
                wups0_e = wups0_t = wups0_c = 0
                for qtype in qtypes:
                    wups0_all += wups0[qtype]
                    wups9_all += wups9[qtype]
                    if qtype[0] == "C":
                        wups0_e += wups0[qtype]
                    if qtype[0] == "T":
                        wups0_t += wups0[qtype]
                    if qtype[0] == "D":
                        wups0_c += wups0[qtype]

                    if num[qtype] != 0:
                        wups0[qtype] = wups0[qtype] / num[qtype]
                        wups9[qtype] = wups9[qtype] / num[qtype]
                    else:
                        wups0[qtype] = 0
                        wups9[qtype] = 0

                # num_e = over_num["C"]
                # num_t = over_num["T"]
                # num_c = over_num["D"]

                # wups0_e /= num_e
                # wups0_t /= num_t
                # wups0_c /= num_c

                wups0_all /= results["ref_num"]
                wups9_all /= results["ref_num"]

                for k in qtypes:
                    wups0[k] = wups0[k] * 100
                    wups9[k] = wups9[k] * 100

                # wups0_e *= 100
                # wups0_t *= 100
                # wups0_c *= 100
                wups0_all *= 100

                return dict(
                    wups0=wups0, wups9=wups9, wups0_all=wups0_all, wups9_all=wups9_all
                )

        return compute_metrics
