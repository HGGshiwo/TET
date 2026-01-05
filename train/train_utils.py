from typing import Optional, Union, Tuple, List, Any, Dict
import time


import torch
from qwen_vl_utils.vision_process import (
    VIDEO_READER_BACKENDS,
    calculate_video_frame_range,
    smart_nframes,
)


def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx)
    video = video.permute(0, 3, 1, 2)  # Convert to TCHW format
    # logger.info(
    #     f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    # )
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps


VIDEO_READER_BACKENDS["decord"] = _read_video_decord
from qwen_vl_utils import process_vision_info


def compress_consecutive_numbers(nums):
    """
    将数字列表中连续的数字转换为起始-结束的格式

    参数:
        nums: 数字列表，要求已排序

    返回:
        转换后的字符串
    """
    if not nums:  # 处理空列表
        return ""
    nums = sorted(nums)
    result = []
    start = nums[0]
    end = nums[0]

    for i in range(1, len(nums)):
        if nums[i] == end + 1:  # 数字连续
            end = nums[i]
        else:  # 数字不连续，记录前一段
            if start == end:  # 单个数字
                result.append(str(start))
            else:  # 连续数字段
                result.append(f"{start}-{end}")
            start = end = nums[i]

    # 处理最后一段
    if start == end:
        result.append(str(start))
    else:
        result.append(f"{start}-{end}")

    return ",".join(result)


from datasets import Dataset
from datasets.table import concat_tables
from datasets.features.features import _align_features
from datasets.arrow_dataset import update_metadata_with_features, update_fingerprint
from datasets.info import DatasetInfo


def concatenate_datasets(
    dsets,
    info=None,
    split=None,
    axis: int = 0,
):
    # Ignore datasets with no rows
    if any(dset.num_rows > 0 for dset in dsets):
        dsets = [dset for dset in dsets if dset.num_rows > 0]
    else:
        # Return first dataset if all datasets are empty
        return dsets[0]

    # Find common format or reset format
    format = dsets[0].format
    if any(dset.format != format for dset in dsets):
        format = {}

    # Concatenate indices if they exist
    indices_table = None

    table = concat_tables([dset._data for dset in dsets], axis=axis)
    if axis == 0:
        features_list = _align_features([dset.features for dset in dsets])
    else:
        features_list = [dset.features for dset in dsets]
    table = update_metadata_with_features(
        table, {k: v for features in features_list for k, v in features.items()}
    )

    # Concatenate infos
    if info is None:
        info = DatasetInfo.from_merge([dset.info for dset in dsets])
    fingerprint = update_fingerprint(
        "".join(dset._fingerprint for dset in dsets),
        concatenate_datasets,
        {"info": info, "split": split},
    )

    # Make final concatenated dataset
    concatenated_dataset = Dataset(
        table,
        info=info,
        split=split,
        indices_table=indices_table,
        fingerprint=fingerprint,
    )
    concatenated_dataset.set_format(**format)
    return concatenated_dataset
