from runner import Runner
import numpy as np
from utils import load_data

iou_rate = []
acc_rate = []
uni_iou_rate = []
uni_acc_rate = []
frame_num = []

def calculate_iou(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)
    # 计算交集
    intersection = set1.intersection(set2)
    # 计算并集
    union = set1.union(set2)
    # 计算IoU
    iou = len(intersection) / len(union) if len(union) != 0 else 0
    return iou

def calculate_acc(list1, list2):
    """
    list1: pred
    list2: truth
    """
    set1 = set(list1)
    set2 = set(list2)
    acc = len(set1.intersection(set2)) / len(set2) if len(set2) != 0 else 0
    return acc

def frame_select(runner, **data):
    if data["qid"] not in select_data2:
        return True
    truth = select_data3[data["qid"]]["answer"]
    if "relevance" in select_data2[data["qid"]]:
        pred = []
        for relevant, idx in zip(select_data2[data["qid"]]["relevance"], select_data2[data["qid"]]["tree_node"]):
            if relevant == 3:
                pred.append(idx)
    elif "answer" in select_data2[data["qid"]]:
        pred = select_data2[data["qid"]]["answer"]
    elif "relevant_idx" in select_data2[data["qid"]]:
        pred = select_data2[data["qid"]]["relevant_idx"]
    uniform = np.linspace(0, select_data[data["qid"]]["last"], len(pred)).astype(int)
    uniform = list(set(uniform))
    iou_rate.append(calculate_iou(pred, truth))
    acc_rate.append(calculate_acc(pred, truth))
    uni_iou_rate.append(calculate_iou(uniform, truth))
    uni_acc_rate.append(calculate_acc(uniform, truth))
    frame_num.append(len(pred))
    return True


if __name__ == "__main__":
    # select_data2 = load_data("./outputs/0413/select2.jsonl") # 模型选
    select_data2 = load_data("./outputs/0329/nextmc_gpt_4o_tree1.jsonl") # video tree 选
    # select_data2 = load_data("./outputs/0508/clip.jsonl") # clip 选
    # select_data2 = load_data("./outputs/0413/select.jsonl") # yolo 选
    # select_data2 = load_data("./outputs/0524/clip_clip2_lvnet.jsonl") # clip2 选
    # select_data2 = load_data("./outputs/0524/clip_clip_top1.jsonl") # clip top1 选
    # select_data2 = load_data("./outputs/0524/clip_clip_top3.jsonl") # clip top3 选
    # select_data2 = load_data("./outputs/0524/clip_clip_top1_yolo.jsonl")  # clip top1 yolo 选
    # select_data2 = load_data("./outputs/0529/clip_clip_top1_limit.jsonl")  # clip top1 yolo 选
    # select_data2 = load_data("./outputs/0529/clip_clip2_lvnet_limit.jsonl") # clip2 lvnet yolo 选
    # select_data2 = load_data("./outputs/0531/clip_blip_top1_limit.jsonl")  # blip top1 yolo 选
    # select_data2 = load_data("./outputs/0531/clip_blip_top1.jsonl")  # blip top1 yolo 选
    # select_data2 = load_data("./outputs/0601/dino_select.jsonl")  # dino select
    
    select_data3 = load_data("./outputs/0502/relevant.jsonl")
    select_data = load_data("./outputs/0413/select.jsonl")
    runner = Runner(frame_select, iter_key="qid")
    runner()
    print("IoU: ", np.mean(iou_rate))
    print("uniform IoU: ", np.mean(uni_iou_rate))
    print("Acc: ", np.mean(acc_rate))
    print("uniform Acc: ", np.mean(uni_acc_rate))
    print("Frame num: ", np.mean(frame_num))