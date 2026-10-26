"""Train the shared TCMFine matcher or fine-tune TCM-Coarse using Lightning 1.9."""

import argparse
import os
from pathlib import Path


def parse_args(argv=None):
    import pytorch_lightning as pl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--main_cfg_path", default=None)
    parser.add_argument("--data-config", "--data_cfg_path", default=None)
    parser.add_argument("--data-root", default=os.environ.get("TCMNET_DATA_ROOT"))
    parser.add_argument("--output-root", default=os.environ.get("TCMNET_OUTPUT_ROOT", "outputs"))
    parser.add_argument("--exp-name", "--exp_name", default="tcmfine_joint")
    parser.add_argument(
        "--train-scope", choices=("full", "tcm-coarse-only"), default=None,
        help="full trains coarse, TCMFine and subpixel; tcm-coarse-only freezes the shared matcher",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ckpt-path",
        "--ckpt_path",
        default="weights/weights.ckpt",
        help="pretrained matcher initialization; ignored when --resume is supplied",
    )
    parser.add_argument(
        "--resume", default=None, help="resume all training state from this checkpoint"
    )
    parser.add_argument("--disable-ckpt", "--disable_ckpt", action="store_true")
    parser.add_argument("--profiler-name", choices=("inference", "pytorch"), default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        accelerator="auto",
        devices="1",
        max_epochs=30,
        precision=32,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    return parser.parse_args(argv)


def build_config(args):
    """Merge user YAML files over portable training defaults."""
    from configs.default import get_cfg_defaults

    config = get_cfg_defaults()
    resume_scope = None
    config.PAIR_MATCHER.MATCH_COARSE.TCM_COARSE_ENABLE = True
    config.PAIR_MATCHER.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
    config.TRAINER.CANONICAL_LR = 0.008
    config.TRAINER.WARMUP_STEP = 0
    config.TRAINER.MSLR_MILESTONES = [10, 15, 20, 25]
    if args.resume:
        from yacs.config import CfgNode

        from models.checkpoint import read_checkpoint

        saved = read_checkpoint(args.resume).get("training_config")
        if saved:
            saved_config = CfgNode.load_cfg(saved)
            # Checkpoints written before TRAIN_SCOPE was introduced froze all
            # but TCM-Coarse; retain their optimizer parameter set on resume.
            if (
                "TRAIN_SCOPE" not in saved_config.TRAINER
                and saved_config.PAIR_MATCHER.MATCH_COARSE.TCM_COARSE_ENABLE
            ):
                config.TRAINER.TRAIN_SCOPE = "tcm-coarse-only"
            config.merge_from_other_cfg(saved_config)
            resume_scope = config.TRAINER.TRAIN_SCOPE
    for path in (args.config, args.data_config):
        if path:
            config.merge_from_file(path)
    if args.train_scope is not None:
        config.TRAINER.TRAIN_SCOPE = args.train_scope
    if resume_scope is not None and config.TRAINER.TRAIN_SCOPE != resume_scope:
        raise ValueError("Cannot change TRAINER.TRAIN_SCOPE when resuming optimizer state")
    if args.data_root:
        config.DATASET.TRAIN_DATA_ROOT = args.data_root
    if not config.DATASET.TRAIN_DATA_ROOT:
        raise ValueError("Set --data-root, TCMNET_DATA_ROOT, or DATASET.TRAIN_DATA_ROOT.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers non-negative.")
    return config


def main():
    # Suppress library update checks without modifying tensor repr or GPU visibility.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    args = parse_args()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from data.dataset import MultiSceneDataModule
    from training.runtime import LocalCheckpointIO, build_profiler
    from training.trainer import TCMTrainer

    config = build_config(args)
    pl.seed_everything(config.TRAINER.SEED, workers=True)
    if (
        not args.resume
        and config.TRAINER.TRAIN_SCOPE == "tcm-coarse-only"
        and not args.ckpt_path
    ):
        raise ValueError("TCM-Coarse-only training needs a pretrained matcher (--ckpt-path).")
    logger = TensorBoardLogger(args.output_root, name=args.exp_name, default_hp_metric=False)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if not args.disable_ckpt:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(Path(logger.log_dir) / "checkpoints"),
                filename="{epoch:02d}",
                save_top_k=-1,
                save_last=True,
            )
        )
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=not args.disable_ckpt,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        replace_sampler_ddp=True,
        plugins=[LocalCheckpointIO()],
        profiler=build_profiler(args.profiler_name),
    )
    config.TRAINER.WORLD_SIZE = trainer.num_devices * trainer.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    config.TRAINER.SCALING = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * config.TRAINER.SCALING
    model = TCMTrainer(config, pretrained_ckpt=None if args.resume else args.ckpt_path)
    data = MultiSceneDataModule(args, config)
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
