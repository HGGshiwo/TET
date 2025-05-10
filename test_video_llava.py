from dataclasses import asdict
from model.builder import build_model
from dataset.data_collator import get_data_collator
from runner.generate_eval_trainer import GenerateEvalTrainer
from dataset.builder import build_dataset
from args import TestingArguments
from transformers import HfArgumentParser

def eval():
    args = HfArgumentParser(TestingArguments).parse_args_into_dataclasses()[0]
    model, tokenizer, processor = build_model(is_training=False, **asdict(args))
    eval_dataset = build_dataset(
        args.dataset_config, args.eval_dataset, is_training=False
    )
    kwargs = {"tokenizer": tokenizer}
    if args.model_type != "long_video":
        kwargs["image_processor"] = processor
        kwargs["select_type"] = args.select_type
        kwargs["delay_load"] = args.delay_load
        kwargs['frame_limit'] = args.frame_limit
    else:
        kwargs['mm_spatial_pool_mode'] = model.config.mm_spatial_pool_mode
        kwargs['mm_spatial_pool_stride'] = model.config.mm_spatial_pool_stride
        kwargs['is_training'] = False
    data_collator = get_data_collator(args.model_type, **kwargs)

    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = GenerateEvalTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if eval_dataset is not None:
        metrics = {}
        for name, dataset in eval_dataset.items():
            trainer.compute_metrics = dataset.get_compute_metrics(tokenizer)
            result = trainer.evaluate(eval_dataset=dataset)
            metrics[name] = result
        print(metrics)


if __name__ == "__main__":
    eval()
