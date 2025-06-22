# 2. 给出起始帧描述，然后让模型给出问题
from runner import Runner
from pathlib import Path
from task_utils import create_model, get_frame
from utils import load_data
import torch

def SortSimilarity(simmat, keywords, maximgslen):
    """
    simmat: (image, tokens, keywords)
    nimgtokens: number of image tokens
    """
    nimgtokens = simmat.shape[1]
    simmat = simmat.flatten(0, 1).T  # (nimages*nimgtokens, nkeywords)
    sort_simmat, sort_idx = torch.sort(simmat, dim=-1, descending=True)
    sort_idx = torch.floor(sort_idx/nimgtokens).to(int)

    curimgslen = 0

    imgidx_kw_dict = dict()
    numrow, numcol = sort_simmat.shape
    
    row_col_list = [0 for _ in range(numrow)]
    token = True

    while token:
        j = 0
        while j < numrow:
            k = 0
            i = row_col_list[j]

            while k < numcol-i:
                col_idx = i+k
                k += 1

                simvalue = sort_simmat[j, col_idx].item()
                img_idx = sort_idx[j, col_idx].item()

                curr_keyword = keywords[j]
                # curr_kfpath = nframes_paths[img_idx]

                if img_idx in imgidx_kw_dict: continue

                else:
                    imgidx_kw_dict[img_idx] = {"kw": curr_keyword, "simvalue": simvalue, "kw_others": []}
                    curimgslen += 1

                    row_col_list[j] = col_idx + 1
                    if curimgslen == maximgslen: return imgidx_kw_dict
                    else: break

            j += 1

        if sum(row_col_list) >= numrow*(numcol-1): token = False

def frame_select(runner, **data):
    image_rate = 0.5
    qid = data["qid"]
    video_path = Path(runner.dataset.config.video_path).joinpath(data["video_path"])
    video = get_frame(video_path, 1)
    if after_yolo:
        video = [video[i] for i in select_data1[qid]["relevant_idx"]]
    obj = select_data[qid]["obj"]
    sim = model.forward(obj, video)
    other = {}
    if not limit_num:
        image_num = int(image_rate * sim.shape[0])
    else:
        image_num = len(select_data2[qid]["answer"])
    if relevant_type.startswith("top"):
        n = int(relevant_type[3:])
        # sim = sim.max(dim=-1).values
        sim, _ = torch.sort(sim, dim=-1, descending=True)
        sim = sim[:, :n].sum(dim=-1)
        sim = torch.topk(sim, image_num, dim=0).indices
        selected = [int(i) for i in sim.cpu().numpy()]
        selected = sorted(selected)
    elif relevant_type == "lvnet":
        out = SortSimilarity(sim, obj, image_num)
        selected = sorted(list(out.keys()))
        other["keyword"] = [out[key]["kw"] for key in selected]
        other["score"] = [out[key]["simvalue"] for key in selected]
    return {"qid": qid, "relevant_idx": selected, "last": len(video), **other}


if __name__ == "__main__":
    # exp_name = "0508"
    # exp_name = "0524"
    # exp_name = "0529"
    exp_name = "0531"
    
    model_name = "clip2"
    # model_name = "clip"
    # model_name = "blip"
    
    # relevant_type = "lvnet"
    relevant_type = "top1"
    # relevant_type = "top3"
    
    after_yolo = False # 在yolo之后进行clip选帧
    limit_num = False  # 使用人工选帧的结果作为数量限制
    
    
    end = "_yolo" if after_yolo else ""
    end = "_limit" if limit_num else end
    output_path = f"./outputs/{exp_name}/clip_{model_name}_{relevant_type}{end}.jsonl"
    data_path = "./outputs/0508/object.jsonl"
    select_data = load_data(data_path)
    
    if after_yolo:
        select_data1 = load_data("./outputs/0413/select.jsonl")
    
    if limit_num:
        select_data2 = load_data("./outputs/0502/relevant.jsonl")
        
    model = create_model(model_name)
    
    runner = Runner(frame_select, output_path, iter_key="qid")
    runner()
