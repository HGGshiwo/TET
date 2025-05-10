from dataclasses import asdict

from model.builder import build_model
from dataset.data_collator import get_data_collator
from runner.generate_eval_trainer import GenerateEvalTrainer
from dataset.builder import build_dataset
from args import LongVideoTraningArguments
from transformers import HfArgumentParser

def train():
    args = HfArgumentParser(LongVideoTraningArguments).parse_args_into_dataclasses()[0]
    kwargs = asdict(args)
    llm_pretrained = kwargs.pop("llm_pretrained")
    model_type = kwargs.pop("model_type")
    kwargs["is_training"] = True
    model, tokenizer, _ = build_model(llm_pretrained, model_type, **kwargs)
    train_dataset = build_dataset(
        args.dataset_config, args.train_dataset, is_training=True
    )
    eval_dataset = build_dataset(
        args.dataset_config, args.eval_dataset, is_training=False
    )
    data_collator = get_data_collator(
        args.model_type,
        tokenizer=tokenizer,
        mm_spatial_pool_stride=model.config.mm_spatial_pool_stride,
        mm_spatial_pool_mode=model.config.mm_spatial_pool_mode,
    )

    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = GenerateEvalTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()

    if eval_dataset is not None:
        metrics = {}
        for name, dataset in eval_dataset.items():
            trainer.compute_metrics = dataset.get_compute_metrics(tokenizer=tokenizer)
            result = trainer.evaluate(eval_dataset=dataset)
            metrics[name] = result
        print(metrics)


if __name__ == "__main__":
    train()
