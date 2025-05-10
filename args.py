from dataclasses import dataclass
from transformers import TrainingArguments
from dataclasses import field


@dataclass
class ExtractingArguments:
    image_size: int = 224
    dataset_config: str = "configs/dataset.yml"
    dataset_name: str = "egoschema"
    model_name: str = "openai/clip-vit-large-patch14"
    batch_size: int = 64 # too small may lead to memory leak
    from_scratch: bool = False


@dataclass
class GeneratingArguments:
    dataset_name: str = "egoschema"
    dataset_config: str = "configs/dataset.yml"
    model_name: str = "gpt-4-1106"
    max_cluster_num: int = 64
    frame_per_cluster: int = 10
    iter_threshold: int = 4
    default_adpative_rate: int = 2
    batch_size: int = 8
    save_every: int = 100


@dataclass
class LongVideoTraningArguments(TrainingArguments):
    train_dataset: list[str] = None
    eval_dataset: list[str] = None
    dataset_config: str = "configs/dataset.yml"
    llm_pretrained: str = "D:/models/LLaVA-Video-7B-Qwen2"
    lora_modules: str = (
        "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    )
    # finetune_modules: list[str] = field(
    #     default_factory=lambda: ["model.mm_projector", "model.image_newline"]
    # )
    logging_steps: int = 10
    finetune_modules: list[str] = field(
        default_factory=lambda: ["model.judge_head"]
    )
    lora_r: int = 128
    lora_alpha: int = 256
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = "outputs/debug"
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    num_train_epochs: int = 3
    model_type: str = "long_video"
    per_device_eval_batch_size: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    frame_limit: int = 16
    delay_load: bool = True
    select_type: str = "select"
    
@dataclass
class TestingArguments(TrainingArguments):
    llm_pretrained: str = "D:/models/LLaVA-Video-7B-Qwen2"
    model_type: str = "long_video"
    eval_dataset: list[str] = None
    dataset_config: str = "configs/dataset.yml"
    attn_implementation: str = "flash_attention_2"
    output_dir: str = "outputs/debug"
    remove_unused_columns: bool = False
    # dataloader_pin_memory: bool = False
    per_device_eval_batch_size: int = 1
    per_device_train_batch_size: int = 1
    delay_load: bool = True
    include_for_metrics: list[str] = field(default_factory=lambda: ["inputs", "loss"])
    batch_eval_metrics: bool = True
    dataloader_num_workers: int = 0
    frame_limit: int = 16
    select_type: str = "select"
