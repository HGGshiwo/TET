from dataset.builder import build_dataset
from args import ExtractingArguments
from transformers import HfArgumentParser
from pathlib import Path
import decord
import asyncio
from data_generation.kmeans_pytorch import kmeans

decord.bridge.set_bridge("torch")
import torch
from tqdm import tqdm
from task_utils import api_forward
from utils import load_data, load_frame_features
import re
from runner import async_run_task


def update_relevance_response(text):
    response = text
    # print("response",response)
    if response is None:
        return None
    relevance_match = re.search(
        r"frame relevance: \[([0-9, ]+)\]", response, flags=re.IGNORECASE
    )
    if relevance_match:
        # Convert the matched string to a list of integers
        relevance = list(map(int, relevance_match.group(1).split(",")))
        return relevance


def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    # Dictionary to store the indices of the closest points for each cluster
    closest_points_idx_per_cluster = {
        cluster_id: [] for cluster_id in range(len(cluster_centers))
    }

    # Iterate over each cluster
    for cluster_id in range(len(cluster_centers)):
        # Filter points belonging to the current cluster
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster = x[indices_in_cluster]

        # Calculate distances from points in the cluster to the cluster center
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)

        if distances.numel() > 0:

            # Find the index (within the cluster) of the point closest to the cluster center
            closest_idx_in_cluster = torch.argmin(distances).item()

            # Map back to the original index in x
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item()

            # Store the global index
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx)

    return closest_points_idx_per_cluster


async def run_task(data, narration):
    prompt1 = "You are presented with a textual description of a first view video clip, it consists of N sparsely sampled from the video The ultimate goal is to answer a question related to this video, choosing the correct option out of five possible answers. Please provide the answer with a single-letter (A, B, C, D, E). It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response. After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are uncertain about the correct option, select the one that seems closest to being correct. Meanwhile, could you provide a relevance score for each frame caption to evaluate their relevance with the query-answering process. The score is between 1,2,3, where 1 indicates low relevance and 3 signifies high relevance. Please return the relevance score in the format of a list of N scores. \n\nDescription: $narration \n\n###\n\n Questions: $question \n Options: \n A: $optionA \n B: $optionB \n C: $optionC \n D: $optionD \n E: $optionE \n\n###\n\n The prediction, explanation, confidence is (please response in the format of 'prediction: \n explanation: \n confidence: \n frame relevance: \n'):"
    prompt1 = prompt1.replace("$narration", narration)
    qusetion = data["question"].split("A. ")[0]
    prompt1 = prompt1.replace("$question", qusetion)
    prompt1 = prompt1.replace("$optionA", data["cm_a0"])
    prompt1 = prompt1.replace("$optionB", data["cm_a1"])
    prompt1 = prompt1.replace("$optionC", data["cm_a2"])
    prompt1 = prompt1.replace("$optionD", data["cm_a3"])
    prompt1 = prompt1.replace("$optionE", data["cm_a4"])
    pre_q = await api_forward(prompt1)
    return prompt1, pre_q


async def build_tree(**data):
    max_cluster_num = 32
    init_cluster_num = 8
    iter_threshold = 4
    default_adpative_rate = 2

    feature_path = "./outputs/nextqa_features"
    cluster_num = init_cluster_num
    iter_threshold = iter_threshold
    adaptive_rate = default_adpative_rate

    # load frame features
    frame_feats = load_frame_features(data["vid"], feature_path)
    frame_feats = frame_feats.cuda()

    ### adaptive width expansion
    while True:
        # width expansion
        cluster_ids_x, cluster_centers = kmeans(
            X=frame_feats,
            num_clusters=cluster_num,
            distance="cosine",
            device=torch.device("cuda:0"),
        )
        # send cluster_ids_x to GPU
        cluster_ids_x = cluster_ids_x.to("cuda")
        cluster_centers = cluster_centers.to("cuda")
        closest_points_idx_per_cluster = find_closest_points_per_cluster(
            frame_feats, cluster_ids_x, cluster_centers
        )
        if closest_points_idx_per_cluster is None:
            # print("closest_points_idx_per_cluster is None")
            continue
        tree_node = sorted(
            [
                value
                for sublist in closest_points_idx_per_cluster.values()
                for value in sublist
            ]
        )

        cluster_ids_x = cluster_ids_x.tolist()
        # relevance scoring
        # parts = narration
        # Recombine parts with their separators

        # for i in range(1, len(parts), 2):
        #     if i + 1 < len(parts):
        #         captions.append(parts[i] + parts[i + 1])

        # Extract relevant captions based on loc_pred indices
        caption = [data["caption"][key] for key in sorted(data["caption"].keys())]
        loc_caption = [caption[i] for i in tree_node if i > 0 and i <= len(caption)]

        # Join the relevant captions with "narration" label
        narr = "\n".join([f"{i+1}: {loc}" for i, loc in enumerate(loc_caption)])
        prompt, pred = await run_task(data, narr)

        # the output is the predicted frame relevance
        frame_relevance = update_relevance_response(pred)
        if frame_relevance is None:
            return None
        high_relevance_frame_num = frame_relevance.count(3)

        if high_relevance_frame_num < iter_threshold:
            if (cluster_num * adaptive_rate < max_cluster_num) and (
                frame_feats.shape[0] > cluster_num * adaptive_rate
            ):
                cluster_num = cluster_num * adaptive_rate
            else:
                break
        else:
            break

    return {
        "qid": data["qid"],
        "tree_node": tree_node,
        "cluster_ids_x": cluster_ids_x,
        "relevance": frame_relevance,
        "pred": pred,
        "prompt": prompt,
    }


if __name__ == "__main__":
    output_path = "./outputs/0329/nextmc_gpt_4o_tree1.jsonl"
    input_path = "./outputs/0329/nextmc_gpt_4o.jsonl"

    def update_data_item(data, input_data):
        # update the data item with the input data
        if data["qid"] not in input_data:
            return None
        pred = input_data[data["qid"]]
        data["caption"] = {k: v["caption"] for k, v in pred.items()}
        return data

    asyncio.run(
        async_run_task(
            build_tree,
            output_path=output_path,
            iter_key="qid",
            iter_frame=False,
            input_path=input_path,
            update_data_item=update_data_item,
            video_fps=1,
        )
    )
