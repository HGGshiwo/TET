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

from collections import defaultdict
from typing import Callable, Optional, Union

from peft import PeftModel
import torch
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig as _GRPOConfig
from dataclasses import dataclass, field
from data_utils import process_vision_info

import copy

from train.trainer.reward_model import RewardModel

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
    object_reward_ratio: float = field(
        default=0, metadata={"help": "目标检测命中时的奖励比例"}
    )
    length_reward_ratio: float = field(default=0, metadata={"help": "长度奖励"})
    keyframe_reward_ratio: float = field(default=0, metadata={"help": "关键帧命中奖励"})


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


class Qwen2VLGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_model: RewardModel,
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
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

        self.reward_model = reward_model

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
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        reward_keys = [key for key in inputs[0].keys()]
        reward_kwargs = {
            key: [
                example[key] for example in inputs for _ in range(self.num_generations)
            ]
            for key in reward_keys
        }

        # (B, 5) -> [format_reward, correct_reward, object_reward, keyframe_reward, length_reward]
        completion_length = list(completion_mask.sum(dim=1).cpu().numpy())
        raw_rewards = self.reward_model(completions=completions, completion_length=completion_length, **reward_kwargs)
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

        for i, name in enumerate(["format", "correct", "object", "keyframe", "length"]):
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
