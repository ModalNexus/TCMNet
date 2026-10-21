import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger

from configs.default import lower_config
from models import PairMatcher
from models.checkpoint import matcher_state_dict, read_checkpoint
from training.loss import TCMFineLoss
from training.optimizers import build_optimizer, build_scheduler
from training.runtime import PassThroughProfiler
from training.supervision import compute_supervision_coarse, compute_supervision_tcmfine


class TCMTrainer(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """Build the shared matcher for full or TCM-Coarse-only training."""
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.pair_matcher_cfg = lower_config(_config["pair_matcher"])
        self.profiler = profiler or PassThroughProfiler()

        # Matcher
        self.matcher = PairMatcher(config=_config["pair_matcher"])
        self.loss = TCMFineLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = matcher_state_dict(read_checkpoint(pretrained_ckpt))
            # strict=False so a pretrained checkpoint loads even though the
            # model can have extra tcm_coarse.* params; assert nothing else
            # is missing/unexpected to keep the safety of strict loading.
            result = self.matcher.load_state_dict(state_dict, strict=False)
            missing = [k for k in result.missing_keys if "tcm_coarse" not in k]
            assert not missing, f"Unexpected missing keys when loading pretrained: {missing}"
            assert not result.unexpected_keys, (
                f"Unexpected keys when loading pretrained: {result.unexpected_keys}"
            )
            logger.info(
                f"Load '{pretrained_ckpt}' as pretrained checkpoint "
                f"(TCMCoarse params freshly initialized: "
                f"{len(result.missing_keys)} tensors)"
            )

        # Full training updates coarse, TCMFine and subpixel parameters.
        # TCM-Coarse-only fine-tuning remains an explicit training scope.
        self.train_scope = self.config.TRAINER.TRAIN_SCOPE
        if self.train_scope not in ("full", "tcm-coarse-only"):
            raise ValueError(f"Unknown TRAINER.TRAIN_SCOPE: {self.train_scope}")
        self._tcm_coarse_enabled = bool(self.config.PAIR_MATCHER.MATCH_COARSE.TCM_COARSE_ENABLE)
        if self.train_scope == "tcm-coarse-only":
            if not self._tcm_coarse_enabled:
                raise ValueError("TCM-Coarse-only training requires TCM_COARSE_ENABLE=true")
            self._freeze_all_but_tcm_coarse()

        # Testing
        self.dump_dir = dump_dir

    def _freeze_all_but_tcm_coarse(self):
        """Freeze all matcher parameters and BN statistics except TCMCoarse."""
        for name, p in self.matcher.named_parameters():
            p.requires_grad = "tcm_coarse" in name
        # requires_grad=False does NOT stop BatchNorm running-stat updates, so
        # force every non-TCMCoarse BN into eval mode to keep the backbone frozen.
        for name, m in self.matcher.named_modules():
            if "tcm_coarse" not in name and isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            ):
                m.eval()

    def train(self, mode=True):
        """Re-assert the freeze every time Lightning flips the module to train."""
        super().train(mode)
        if mode and getattr(self, "train_scope", None) == "tcm-coarse-only":
            self._freeze_all_but_tcm_coarse()
        return self

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Save non-tensor TCMCoarse settings for unambiguous inference loading."""
        checkpoint["matcher_config"] = self.pair_matcher_cfg
        checkpoint["training_config"] = self.config.dump()

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs
    ):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * abs(
                    self.config.TRAINER.TRUE_LR - base_lr
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}")

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Pairwise feature matcher"):
            self.matcher(batch)

        with self.profiler.profile("Compute TCMFine supervision"):
            compute_supervision_tcmfine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

        return {
            "loss": batch["loss"],
            "loss_c": batch["loss_c"],
            "loss_tcmfine_window": batch["loss_tcmfine_window"],
            "loss_tcmfine_subpixel": batch["loss_tcmfine_subpixel"],
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_loss_c = torch.stack([x["loss_c"] for x in outputs]).mean()
        avg_tcmfine_window = torch.stack([x["loss_tcmfine_window"] for x in outputs]).mean()
        avg_tcmfine_subpixel = torch.stack([x["loss_tcmfine_subpixel"] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                "train/epoch_avg_loss", avg_loss, global_step=self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "train/epoch_avg_loss_c", avg_loss_c, global_step=self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "train/epoch_avg_loss_tcmfine_window", avg_tcmfine_window,
                global_step=self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "train/epoch_avg_loss_tcmfine_subpixel", avg_tcmfine_subpixel,
                global_step=self.current_epoch
            )

            print(
                f"Epoch {self.current_epoch}: the average loss is "
                f"{avg_loss_c.item():.4f} + {avg_tcmfine_window.item():.4f} "
                f"+ {avg_tcmfine_subpixel.item():.4f} = {avg_loss.item():.4f}"
            )
