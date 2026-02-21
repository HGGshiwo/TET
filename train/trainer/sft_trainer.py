from dataclasses import dataclass
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
from trl import SFTConfig, SFTTrainer
from transformers.trainer import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


@dataclass
class GroupRLSFTConfig(SFTConfig):
    lm_head_rl_rate: int = 2  # lm_head学习率是基础学习率的倍数


class GroupRLSFTTrainer(SFTTrainer):

    def create_optimizer(self):
        """
        Setup the optimizer with different learning rates for lm_head and other parameters.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            # 1. 基础配置：获取权重衰减参数名 + 定义分层学习率（可根据需求调整）
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # 核心：定义lm_head的学习率（建议为其他参数的3-5倍）
            lm_head_lr = self.args.learning_rate * self.args.lm_head_rl_rate  # 可根据实际情况调整倍数
            base_lr = self.args.learning_rate

            # 2. 拆分参数组：
            # - 组1: lm_head参数 + 有权重衰减
            # - 组2: lm_head参数 + 无权重衰减
            # - 组3: 其他参数 + 有权重衰减
            # - 组4: 其他参数 + 无权重衰减
            optimizer_grouped_parameters = [[] for _ in range(4)]
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if "lm_head" in n:
                    idx = 0 if n in decay_parameters else 1
                    optimizer_grouped_parameters[idx].append(p)
                else:
                    idx = 2 if n in decay_parameters else 3
                    optimizer_grouped_parameters[idx].append(p)
            lr = [lm_head_lr, lm_head_lr, base_lr, base_lr]
            weight_decay = [self.args.weight_decay, 0, self.args.weight_decay, 0]
            optimizer_grouped_parameters = [
                dict(lr=_lr, weight_decay=_wd, params=_param)
                for _lr, _wd, _param in zip(
                    lr, weight_decay, optimizer_grouped_parameters
                )
            ]

            for i, g in enumerate(optimizer_grouped_parameters):
                print(f"group{i}: {len(g['params'])}")

            # 过滤空参数组（避免报错）
            optimizer_grouped_parameters = [
                g for g in optimizer_grouped_parameters if len(g["params"]) > 0
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if (
                "bitsandbytes" in str(optimizer_cls)
                and optimizer_kwargs.get("optim_bits", None) == 8
            ):
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
