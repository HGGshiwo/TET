# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import textwrap
from collections import defaultdict
from typing import Any, Callable, List, Optional, Union
import random

from peft import PeftModel
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig as _GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from dataclasses import dataclass, field
from data_utils import process_vision_info

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


# 检查 unsloth 是否可用
def _is_unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class GRPOConfig(_GRPOConfig):
    temporal: bool = field(default=False)
    len_control: bool = field(default=False)
    temperature: float = field(default=1)
    use_vllm: bool = field(default=False, metadata={"help": "是否使用 vllm 加速训练。"})
    use_unsloth: bool = field(
        default=False,
        metadata={
            "help": (
                "是否使用 unsloth 加速训练。启用后将使用 unsloth 的 FastVisionModel 加载模型，"
                "可显著降低显存占用并提升训练速度（需要安装 unsloth 包）。"
                "若 unsloth 未安装，该参数会被忽略并回退到标准加载方式。"
            )
        },
    )
    reward_type: str = field(
        default=None,
        metadata={
            "help": (
                "baseline: 使用默认reward"
                "mode_inner_outer: 例如step_mean_min -> mode=step, inner_agg=mean, outer_agg=min"
            )
        },
    )
    bottom_k_ratio: float = field(
        default=0.2,
        metadata={"help": "当agg为bottom_k的时候, 设置k的大小, 默认后20%"},
    )
    alpha: float = field(default=0.5, metadata={"help": "置信度奖励的权重超参"})
    reward_per_100_tokens: float = field(
        default=0,
        metadata={"help": "每生成100个token奖励的分数，默认为0表示不使用长度奖励"},
    )


def repeat(tensor: torch.Tensor, dim: int, repeat_num: int):
    """
    在dim维度重复repeat_num次
    (2, 2), dim=0, num=4 -> (4, 2, 2) -> (4*2, 2)
    """
    origin_size = list(tensor.size())
    expand_size = copy.deepcopy(origin_size)
    expand_size.insert(dim, repeat_num)

    origin_size[dim] = origin_size[dim] * repeat_num
    return tensor.unsqueeze(dim).expand(expand_size).reshape(origin_size)


class ConfidenceRewardComputer:
    def __init__(
        self,
        split_token_id: int = None,
        step_split_token_id: int = None,
        mode="step",
        inner_agg="mean",
        outer_agg="min",
        bottom_k_ratio=0.2,
        alpha=0.5,
        reward_per_100_tokens=0,
    ):
        """
        mode: 'token' 或 'step'
        inner_agg: 'mean', 'min', 'bottom_k' (Token -> Step)
        outer_agg: 'mean', 'min' (Step -> Reward)
        """
        if alpha != 0:
            print(
                "We are using ConfidenceRewardComputer with mode={}, inner_agg={}, outer_agg={}, bottom_k_ratio={}, alpha={}".format(
                    mode, inner_agg, outer_agg, bottom_k_ratio, alpha
                )
            )
        self.mode = mode
        self.inner_agg = inner_agg
        self.outer_agg = outer_agg
        self.bottom_k_ratio = bottom_k_ratio
        self.alpha = alpha
        self.split_token_id = split_token_id
        self.step_split_token_id = step_split_token_id
        self.reward_per_100_tokens = reward_per_100_tokens

    def _aggregate(self, logprobs_tensor, method):
        """核心聚合算子"""
        if len(logprobs_tensor) == 0:
            return 0.0  # 防御性编程

        if method == "mean":
            return logprobs_tensor.mean().item()
        elif method == "min":
            return logprobs_tensor.min().item()
        elif method == "bottom_k":
            k = max(1, int(len(logprobs_tensor) * self.bottom_k_ratio))
            # 取最小的 K 个求均值
            bottom_k_vals, _ = torch.topk(logprobs_tensor, k, largest=False)
            return bottom_k_vals.mean().item()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def compute_reward(self, all_logprobs, step_indices):
        """
        计算单条样本的置信度 Reward
        all_logprobs: Tensor, 形状 [N], 所有输出的对数概率
        step_indices: List[List[int]], 记录每个 step 包含的 token 索引。
                      例如: [[0,1,2,3], [4,5,6], [7,8]]
        """
        # 情况 1: Token 级别 (直接对所有 token 聚合一次)
        if self.mode == "token":
            flat_indices = [idx for step in step_indices for idx in step]
            if not flat_indices:  # 防御：如果全是空的
                return 0.0

            # all_logprobs 是 1D Tensor，直接用列表做索引提取
            valid_logprobs = all_logprobs[flat_indices]
            return self._aggregate(valid_logprobs, self.inner_agg)

        # 情况 2: Step 级别 (两阶段聚合)
        elif self.mode == "step":
            step_scores = []

            # Phase 1: Inner Aggregation (Token -> Step)
            for indices in step_indices:
                if len(indices) == 0:
                    continue
                step_logprobs = all_logprobs[indices]
                step_score = self._aggregate(step_logprobs, self.inner_agg)
                step_scores.append(step_score)

            if not step_scores:
                return 0.0  # 防御性编程

            step_scores_tensor = torch.tensor(step_scores)

            # Phase 2: Outer Aggregation (Step -> Reward)
            final_reward = self._aggregate(step_scores_tensor, self.outer_agg)
            return final_reward

        else:
            raise ValueError("Mode must be 'token' or 'step'")

    def get_step_indices(self, completion_ids: torch.Tensor):
        """获取每个step的token索引, 输入[0, 1, 2, 3, step, 4, 5, 6, step, 7, think, ..., think]
        输出: [[0, 1, 2, 3], [4, 5, 6], [7]]"""
        split_token_id = self.split_token_id
        step_split_token_id = self.step_split_token_id

        # 找到 split_token 出现的位置
        split_positions = (completion_ids == split_token_id).nonzero(as_tuple=True)[0]
        # 确定第一个部分的范围（从开头到第一个 split_token 之前）
        if len(split_positions) != 2:
            return None
        else:
            think_start, think_end = 0, split_positions[0].item()

        # 提取第一个部分的 token ids
        think_ids = completion_ids[think_start:think_end]

        # 在第一个部分内找到 step_split_token 的位置
        step_positions = (think_ids == step_split_token_id).nonzero(as_tuple=True)[0]

        # 切分得到该样本的 step indices（绝对索引）
        sample_step_indices = []
        prev = 0
        for pos in step_positions:
            pos_item = pos.item()
            if prev < pos_item:  # 非空 step
                step_indices = list(range(think_start + prev, think_start + pos_item))
                sample_step_indices.append(step_indices)
            prev = pos_item + 1
        # 处理最后一个 step
        if prev < len(think_ids):
            step_indices = list(range(think_start + prev, think_start + len(think_ids)))
            sample_step_indices.append(step_indices)

        return sample_step_indices

    def __call__(
        self,
        batch_completion_ids: torch.Tensor,
        batch_completion_logps: torch.Tensor,
        correct_flags: List[int],
        format_flags: List[int],
        completion_mask: torch.Tensor,
    ):
        """
        计算整个 batch 的置信度奖励
        batch_completion_ids: Tensor, 形状 [B, N], 包含 B 条样本的 token ids
        batch_completion_logprobs: Tensor, 形状 [B, N], 包含对应的对数概率

        retrun: Tensor 形状[B, 4]:
        [format_reward, correct_reward, confidence_reward, length_reward]
        """

        if self.alpha != 0:
            rewards = []

            for completion_ids, logprobs, mask in zip(
                batch_completion_ids, batch_completion_logps, completion_mask
            ):
                mask = mask.bool()
                logprobs = logprobs[mask]  # 只保留有效 token 的 logprobs
                completion_ids = completion_ids[mask]

                step_indices = self.get_step_indices(completion_ids)
                if step_indices is None:
                    rewards.append(0.0)  # 格式错误的样本奖励为0
                else:
                    reward = self.compute_reward(logprobs, step_indices)
                    rewards.append(reward)

            conf_scores = torch.tensor(rewards)

            # 1. 组内 Z-Score 归一化 (让置信度相对化，高于平均的为正，低于平均的为负)
            mean_conf = conf_scores.mean()
            std_conf = conf_scores.std() + 1e-8
            normalized_conf = (conf_scores - mean_conf) / std_conf
        else:
            normalized_conf = torch.zeros(
                len(batch_completion_ids)
            )  # 全零，避免后续计算错误

        # 2. 计算最终 Reward

        final_rewards = []

        for i in range(len(correct_flags)):
            final_reward = [0 for i in range(4)]
            final_reward[0] = -0.5 if format_flags[i] == 0 else 0
            final_reward[1] = 0 if correct_flags[i] == 0 else 1
            if correct_flags[i] == 1:
                # 答对了：置信度加成 (笃定且正确奖励更高，心虚但蒙对的奖励降低)
                # 不管怎样不给负的奖励
                r = self.alpha * torch.max(torch.tensor(0.0), normalized_conf[i]).item()
            else:
                if normalized_conf[i] > 0:
                    # 答错了但很自信：给轻微惩罚，鼓励模型修正过于自信的错误
                    r = -0.1 * normalized_conf[i].item()
                else:
                    # 答错了但是不自信: 0分
                    r = 0.0
            final_reward[2] = r
            if self.reward_per_100_tokens is not None:
                final_reward[3] = self.reward_per_100_tokens * (
                    len(completion_ids) / 100
                )

            final_rewards.append(final_reward)

        return final_rewards


class Qwen2VLGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        check_correct_func: Callable,
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        compute_metrics: Callable = None,
        accuracy_compare_func: Callable = None,
        split_token_id: int = None,
        step_split_token_id: int = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        if args.train_batch_size != 1:
            raise ValueError(
                f"{self.__class__.__name__} only support train_batch_size==1!"
            )

        if args.reward_type is not None:
            assert (
                split_token_id is not None
            ), "reward_type is not None, but split_token_id is None!"
            assert (
                step_split_token_id is not None
            ), "reward_type is not None, but step_split_token_id is None!"
            assert args.alpha != 0, "reward_type is not None, but alpha = 0!"

            mode, inner_agg, outer_agg = args.reward_type.split("_")
        else:
            mode, inner_agg, outer_agg = None, None, None
        self.cfc = ConfidenceRewardComputer(
            split_token_id=split_token_id,
            step_split_token_id=step_split_token_id,
            mode=mode,
            inner_agg=inner_agg,
            outer_agg=outer_agg,
            bottom_k_ratio=args.bottom_k_ratio,
            alpha=args.alpha,
            reward_per_100_tokens=args.reward_per_100_tokens,
        )

        # 模型初始化参数优化：启用flash attention减少显存，控制cache使用
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        self._use_unsloth = False  # 实际是否使用了 unsloth
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )

            # unsloth 加速加载路径
            if args.use_unsloth and _is_unsloth_available():
                from unsloth import FastVisionModel

                # unsloth FastVisionModel 不接受这些参数，quantization_config/device_map 由 unsloth 自己管理
                _unsloth_exclude = {
                    "attn_implementation",
                    "use_cache",
                    "device_map",
                    "quantization_config",
                }
                unsloth_kwargs = {
                    k: v
                    for k, v in model_init_kwargs.items()
                    if k not in _unsloth_exclude
                }
                load_in_4bit = unsloth_kwargs.pop("load_in_4bit", True)
                padding_side = processing_class.tokenizer.padding_side
                self._use_vllm = args.use_vllm
                model, processing_class = FastVisionModel.from_pretrained(
                    model_name=model_id,
                    load_in_4bit=load_in_4bit,
                    max_seq_length=2048,
                    fast_inference=args.use_vllm,
                    use_gradient_checkpointing=(
                        "unsloth" if args.gradient_checkpointing else False
                    ),
                    gpu_memory_utilization=0.5,  # vllm KV cache 预占比例，48GB GPU 下约 24GB
                    **unsloth_kwargs,
                )
                self._use_unsloth = True
                import warnings

                warnings.warn(
                    "Unsloth FastVisionModel 已加载，processing_class 由 unsloth 提供，"
                    "外部传入的 processing_class 参数将被忽略。",
                    UserWarning,
                )
                processing_class.tokenizer.padding_side = padding_side
            elif args.use_unsloth and not _is_unsloth_available():
                import warnings

                warnings.warn(
                    "use_unsloth=True 但未检测到 unsloth 包，回退到标准加载方式。"
                    "请通过 `pip install unsloth` 安装。",
                    UserWarning,
                )
                # 回退到标准加载
                if "Qwen2-VL" in model_id:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
                elif "Qwen2.5-VL" in model_id:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
                elif "Aria" in model_id:
                    model_init_kwargs.pop("use_cache")
                    model = AriaForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
            else:
                # 按模型类型加载，避免冗余判断
                if "Qwen2-VL" in model_id:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
                elif "Qwen2.5-VL" in model_id:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
                elif "Aria" in model_id:
                    model_init_kwargs.pop("use_cache")
                    model = AriaForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model, **model_init_kwargs
                    )
        else:
            if args.use_unsloth:
                raise ValueError("use_unsloth = True时只支持传递模型路径!")
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # 参考模型初始化：仅在必要时创建，减少显存占用
        self.ref_model = None

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()  # 开启 checkpoint
            model.enable_input_require_grads()  # 【关键】强制让输入层输出需要梯度

        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif not isinstance(model, PeftModel) and (peft_config is None):
            # 非 LoRA 全量模型：无法通过 disable_adapter() 获取初始权重，需独立创建 ref_model
            self.ref_model = create_reference_model(model)

        # 处理类初始化：复用pad/eos token id，避免重复计算
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
        pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.pad_token_id = pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

        self.check_correct = check_correct_func

        # 数据collator保持原有逻辑
        def data_collator(features):
            return features

        # 生成配置初始化：复用基础配置，减少冗余定义
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temporal = args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=args.temperature,
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        self.len_control = args.len_control
        self.beta = args.beta
        self.accuracy_compare_func = accuracy_compare_func

        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_metrics=compute_metrics,
        )

        self.model_accepts_loss_kwargs = False

        # 参考模型和奖励函数加速准备：仅必要时prepare，减少显存
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, **kwargs).logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        selected_logits = torch.gather(
            logits, dim=2, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
        log_sum_exp = logits.logsumexp(dim=-1)
        per_token_logps = selected_logits - log_sum_exp

        del logits, selected_logits, log_sum_exp
        return per_token_logps

    def remove_none_from_data(self, data):
        # 原地清理None值，避免创建新列表，减少显存
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _prepare_inputs(self, inputs):
        return inputs

    def _prepare_prompt_inputs(self, inputs, do_shuffle=False):
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        input_copy = [inputs[i]["prompt"] for i in range(len(inputs))]
        input_copy = self.remove_none_from_data(input_copy)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            input_copy, return_video_kwargs=True
        )
        if do_shuffle:
            if not video_inputs:
                return None
            indices = [
                torch.randperm(video_inputs[i].size(0))
                for i in range(len(video_inputs))
            ]
            video_inputs = [video_inputs[i][indice] for i, indice in enumerate(indices)]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        return prompt_inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.accuracy_compare_func is None:
            raise RuntimeError("accuracy_compare_func is None!")
        prompt_inputs = self._prepare_prompt_inputs(inputs)
        prompt_ids = prompt_inputs["input_ids"]
        eval_gen_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=1024,
            pad_token_id=self.processing_class.tokenizer.pad_token_id,
            use_cache=True,
        )

        with torch.no_grad():
            with unwrap_model_for_generation(
                model, self.accelerator
            ) as unwrapped_model:
                generated_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=eval_gen_config,
                    use_model_defaults=False,  # 不使用模型默认值, 否则可能会覆盖generation_config
                )
            completion_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(prompt_ids, generated_ids)
            ]
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

        predictions = []
        for completion, example in zip(completions, inputs):
            truth = example.get("truth")
            if truth is not None:
                is_correct = self.accuracy_compare_func(completion, truth)
                predictions.append(1.0 if is_correct else 0.0)
            else:
                predictions.append(0.0)

        labels = torch.tensor(
            predictions, dtype=torch.float32, device=self.accelerator.device
        )
        loss = torch.zeros_like(labels)
        logits = torch.zeros_like(labels)
        return (loss, logits, labels)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompts = [item["prompt"] for item in inputs]
        prompt_inputs = self._prepare_prompt_inputs(inputs)
        prompt_ids = prompt_inputs["input_ids"]

        # 时序相关处理：仅在有视频输入时处理，避免冗余计算
        shuffled_prompt_ids, shuffled_completion_ids = None, None
        if self.temporal:
            shuffled_prompt_inputs = self._prepare_prompt_inputs(
                inputs, do_shuffle=True
            )
            if shuffled_prompt_inputs is not None:
                shuffled_prompt_ids = shuffled_prompt_inputs["input_ids"]

        # 生成补全序列：使用unwrap_model_for_generation减少显存占用
        # 若启用 unsloth，直接使用模型的 fast_generate 接口以获得更高吞吐
        if self._use_unsloth:
            from unsloth import FastVisionModel

            # 切换到推理模式：恢复 use_cache=True，关闭 gradient checkpointing
            FastVisionModel.for_inference(model)
            if self._use_vllm:
                # vllm 不接受 input_ids 张量，需要传 token id 列表
                input_ids_list = prompt_inputs["input_ids"].tolist()
                prompt_completion_ids = model.fast_generate(
                    input_ids_list, generation_config=self.generation_config
                )
            else:
                prompt_completion_ids = model.fast_generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # prompt_completion_ids 在 logps 计算完后释放

            if shuffled_prompt_ids is not None:
                if self._use_vllm:
                    shuffled_input_ids_list = shuffled_prompt_inputs[
                        "input_ids"
                    ].tolist()
                    shuffled_prompt_completion_ids = model.fast_generate(
                        shuffled_input_ids_list,
                        generation_config=self.shuffled_generation_config,
                    )
                else:
                    shuffled_prompt_completion_ids = model.fast_generate(
                        **shuffled_prompt_inputs,
                        generation_config=self.shuffled_generation_config,
                    )
                shuffled_prompt_length = shuffled_prompt_ids.size(1)
                shuffled_completion_ids = shuffled_prompt_completion_ids[
                    :, shuffled_prompt_length:
                ]
                del shuffled_prompt_completion_ids
            # 切换回训练模式：关闭 use_cache，恢复 gradient checkpointing
            FastVisionModel.for_training(model)
        else:
            with unwrap_model_for_generation(
                model, self.accelerator
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    use_model_defaults=False,  # 不使用模型默认值, 否则可能会覆盖generation_config
                )

                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]

                if shuffled_prompt_ids is not None:
                    shuffled_prompt_completion_ids = unwrapped_model.generate(
                        **shuffled_prompt_inputs,
                        generation_config=self.shuffled_generation_config,
                        use_model_defaults=False,
                    )
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_completion_ids = shuffled_prompt_completion_ids[
                        :, shuffled_prompt_length:
                    ]
                    del shuffled_prompt_completion_ids

        # 生成补全掩码：控制张量大小，避免冗余
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # 释放临时张量
        del is_eos, eos_idx, sequence_indices

        # repeat_num 为生成的序列总数（= batch_size * num_generations），用于将视觉张量与每条生成序列对应
        repeat_num = completion_ids.size(0)
        completion_ids_len = completion_ids.size(1)

        # 处理视觉输入张量：使用expand代替repeat（视图复用），减少显存
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        data_type = inputs[0]["data_type"]

        if data_type == "image":
            # expand代替repeat，减少数据复制
            prompt_inputs["pixel_values"] = repeat(
                prompt_inputs["pixel_values"], 0, repeat_num
            )
            prompt_inputs["image_grid_thw"] = repeat(
                prompt_inputs["image_grid_thw"], 0, repeat_num
            )
        elif data_type == "video":
            prompt_inputs["pixel_values_videos"] = repeat(
                prompt_inputs["pixel_values_videos"], 0, repeat_num
            )
            prompt_inputs["video_grid_thw"] = repeat(
                prompt_inputs["video_grid_thw"], 0, repeat_num
            )
            if "second_per_grid_ts" in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]

        # 计算token级log概率：捕获异常并释放临时张量
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, **prompt_inputs
        )
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        # 计算参考模型log概率：使用推理模式，减少显存
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, **prompt_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, **prompt_inputs
                    )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        # 释放 prompt_completion_ids、prompt_inputs（含大型视觉张量）和 prompt_ids，后续不再需要
        del prompt_completion_ids, prompt_inputs, prompt_ids

        # 计算KL散度：裁剪数值范围，减少异常值，同时释放临时张量
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        del x_clamped, ref_per_token_logps

        # 解码补全序列并计算奖励：推理模式减少显存
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        input_truth = [
            example["truth"] for example in inputs for _ in range(self.num_generations)
        ]

        correct_flags, format_flags = self.check_correct(
            completions=completions, truth=input_truth
        )
        # (B, 4) -> [format_reward, correct_reward, confidence_reward, length_reward]
        raw_rewards = self.cfc(
            batch_completion_ids=completion_ids,
            batch_completion_logps=per_token_logps,
            correct_flags=correct_flags,
            format_flags=format_flags,
            completion_mask=completion_mask,
        )
        rewards = torch.tensor(raw_rewards, device=device, dtype=torch.float32)
        # 释放临时变量
        del completions, prompts, completion_ids

        # 奖励求和：控制数据类型，减少显存
        rewards_sum = rewards.sum(dim=1)

        # 计算分组奖励：复用张量形状，减少拷贝
        grouped_shape = (-1, self.num_generations)
        mean_grouped_rewards = rewards_sum.view(grouped_shape).mean(dim=1)
        std_grouped_rewards = rewards_sum.view(grouped_shape).std(dim=1)

        # 归一化优势值：使用expand代替repeat_interleave，减少数据复制
        mean_grouped_rewards = mean_grouped_rewards.expand(
            self.num_generations, -1
        ).reshape(-1)
        std_grouped_rewards = std_grouped_rewards.expand(
            self.num_generations, -1
        ).reshape(-1)
        advantages = (rewards_sum - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        del mean_grouped_rewards, std_grouped_rewards

        # 计算损失：控制计算图，减少显存
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        # 释放所有临时张量，最大化显存释放
        del per_token_loss, advantages, per_token_kl

        # 日志指标计算：使用accelerator.gather_for_metrics，减少显存
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        completion_length = completion_length.float().mean().item()

        self._metrics["completion_length"].append(completion_length)
        gather_rewards = self.accelerator.gather_for_metrics(rewards)
        reward_per_func = gather_rewards.mean(0)

        for i, name in enumerate(["format", "correct", "confidence", "length"]):
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())

        all_reward_sum = gather_rewards.sum(dim=1)
        self._metrics["reward"].append(all_reward_sum.mean().item())
        self._metrics["reward_std"].append(all_reward_sum.std().item())

        # 释放剩余临时张量
        del (
            gather_rewards,
            all_reward_sum,
            reward_per_func,
            rewards,
            per_token_logps,
            completion_mask,
        )

        torch.cuda.empty_cache()
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
