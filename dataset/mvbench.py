from .builder import BaseDataset
from utils import *
import json

OPTIONS = ["A", "B", "C", "D", "E", "F"]


class MVBench(BaseDataset):
    
    data_list = {
        "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", "perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", "perception/videos/", "video", False),
        # "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
        "Character Order": ("character_order.json", "perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
        # "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
    }

    def __init__(self, config, split="test"):
        super().__init__(config, split)
        
    def get_anno(self):
        data_list = []
        data_dir = self.config.anno_path
        for k, v in self.data_list.items():
            idx = 0
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    **data, # "question", "candidates", "answer", "video"
                    "qid": f"{k}_{idx}",
                })
                idx += 1
        return data_list
        
    def build(self):
        data = []
        for item in self.anno:
            qid = item['qid']
            question = item["question"]
            question = question.strip()
            options = []
            for i, cand in enumerate(item['candidates']):
                options.append(f"{OPTIONS[i]}. {str(cand).strip()}")
            question = "\n".join([question] + options)
            new_item = {
                "qid": qid,
                "vid": item['video'].replace(".mp4", ""),
                "video_path": os.path.join(item["prefix"], item['video']),
                "question": question,
                "options": options,
            }
            if item["bound"]:
                new_item["video_start"] = item["start"]
                new_item["video_end"] = item["end"]
            if item.get("answer", None) is not None:
                idx = item["candidates"].index(item["answer"])
                assert idx >= 0 
                new_item["truth"] = OPTIONS[idx]
            data.append(new_item)
        return data
