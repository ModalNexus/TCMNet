"""Shared configuration for TCMNet, TCM-Coarse and TCM-Fine."""

from yacs.config import CfgNode
from yacs.config import CfgNode as CN


def lower_config(config):
    """Convert a YACS configuration to nested dictionaries with lowercase keys."""
    if not isinstance(config, CfgNode):
        return config
    return {key.lower(): lower_config(value) for key, value in config.items()}


def get_cfg_defaults():
    """Return a fresh configuration; loading one experiment never mutates another."""
    return CN(
        {
            "PAIR_MATCHER": {
                "RESOLUTION": (8, 2, 1),
                "FINE_WINDOW_SIZE": 5,
                "FINE_CONCAT_COARSE_FEAT": True,
                "MATCH_COARSE": {
                    "THR": 0.3,
                    "BORDER_RM": 2,
                    "DSMAX_TEMPERATURE": 0.1,
                    "TRAIN_COARSE_PERCENT": 0.2,
                    "TRAIN_PAD_NUM_GT_MIN": 200,
                    "TCM_COARSE_ENABLE": False,
                    "TCM_COARSE_WINDOW": 5,
                    "TCM_COARSE_TOPK": 2,
                    "TCM_COARSE_MAX_CANDIDATES": 4096,
                    "TCM_COARSE_CHANNELS": 8,
                    "TCM_COARSE_KERNEL": 3,
                    "TCM_COARSE_ALPHA": 1.0,
                },
                "FINE": {"DSMAX_TEMPERATURE": 0.1, "THR": 0.1},
                "LOSS": {
                    "FINE_TYPE": "focal",
                    "POS_WEIGHT": 1.0,
                    "NEG_WEIGHT": 1.0,
                    "FOCAL_ALPHA": 0.25,
                    "FOCAL_GAMMA": 2.0,
                    "COARSE_WEIGHT": 0.5,
                    "FINE_WEIGHT": 0.3,
                    "SUB_WEIGHT": 0.1,
                },
            },
            "DATASET": {
                "TRAINVAL_DATA_SOURCE": "Multi-Modality",
                "TRAIN_DATA_ROOT": None,
                "IMAGE_SIZE": 512,
                "PAD_IMAGES": True,
                "DIVISIBILITY": 8,
                "AUGMENTATION_TYPE": None,
            },
            "TRAINER": {
                "TRAIN_SCOPE": "full",
                "WORLD_SIZE": 1,
                "TRUE_BATCH_SIZE": None,
                "CANONICAL_BS": 64,
                "CANONICAL_LR": 0.006,
                "SCALING": None,
                "OPTIMIZER": "adamw",
                "TRUE_LR": None,
                "ADAM_DECAY": 0.0,
                "ADAMW_DECAY": 0.1,
                "WARMUP_TYPE": "linear",
                "WARMUP_RATIO": 0.0,
                "WARMUP_STEP": 4800,
                "SCHEDULER": "MultiStepLR",
                "SCHEDULER_INTERVAL": "epoch",
                "MSLR_MILESTONES": [3, 6, 9, 12],
                "MSLR_GAMMA": 0.5,
                "COSA_TMAX": 30,
                "ELR_GAMMA": 0.999992,
                "GRADIENT_CLIPPING": 0.5,
                "SEED": 66,
            },
        }
    )


def get_matcher_config():
    """Return a fresh matcher dictionary; no mutable global config is reused."""
    return lower_config(get_cfg_defaults().PAIR_MATCHER)


default_cfg = get_matcher_config()


DIRECTIONS = {"1->2": "rs", "2->3": "si", "3->1": "ir"}


PAIR_SPECS = {
    "rs": ("modality1", "modality2"),
    "si": ("modality2", "modality3"),
    "ir": ("modality3", "modality1"),
}
