"""Load old pair-matcher and TCM-Coarse checkpoints through one strict path."""

from collections.abc import Mapping

import torch

from configs.default import get_matcher_config


def read_checkpoint(path):
    """Read a trusted local checkpoint, including Lightning training metadata."""
    return normalize_checkpoint(torch.load(path, map_location="cpu", weights_only=False))


def normalize_checkpoint(value):
    """Translate historical names at the checkpoint boundary, without mutation.

    This is the only runtime compatibility map for pre-paper checkpoint names.
    New source, configuration and saved tensors use TCM-Coarse consistently.
    """
    replacements = {
        "SPARC_": "TCM_COARSE_",
        "sparc_": "tcm_coarse_",
        ".sparc.": ".tcm_coarse.",
        "MGDPT_IMG_RESIZE": "IMAGE_SIZE",
        "MGDPT_IMG_PAD": "PAD_IMAGES",
        "MGDPT_DF": "DIVISIBILITY",
    }

    def rename(text):
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            renamed = rename(key) if isinstance(key, str) else key
            if renamed in result:
                raise ValueError(f"Conflicting old/new checkpoint keys: {key}")
            if key == "training_config" and isinstance(item, str):
                import yaml

                result[renamed] = yaml.safe_dump(
                    normalize_checkpoint(yaml.safe_load(item)), sort_keys=False
                )
            else:
                result[renamed] = normalize_checkpoint(item)
        return result
    if isinstance(value, list):
        return [normalize_checkpoint(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_checkpoint(item) for item in value)
    return value


def matcher_state_dict(checkpoint):
    checkpoint = normalize_checkpoint(checkpoint)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must contain a model state dictionary.")
    return {
        (key[len("matcher.") :] if key.startswith("matcher.") else key): value
        for key, value in state.items()
    }


def checkpoint_config(checkpoint):
    """Recover saved settings, or infer TCMCoarse tensor shapes for old checkpoints."""
    from copy import deepcopy

    checkpoint = normalize_checkpoint(checkpoint)
    config = deepcopy(checkpoint.get("matcher_config", get_matcher_config()))
    state = matcher_state_dict(checkpoint)
    tcm_coarse_keys = [key for key in state if key.startswith("coarse_matching.tcm_coarse.")]
    config["match_coarse"]["tcm_coarse_enable"] = bool(tcm_coarse_keys)
    if tcm_coarse_keys:
        weight = state["coarse_matching.tcm_coarse.conv.0.convs.0.weight"]
        config["match_coarse"]["tcm_coarse_channels"] = int(weight.shape[0])
        config["match_coarse"]["tcm_coarse_kernel"] = int(weight.shape[-1])
    return config


def load_pair_matcher(path, *, coarse_threshold=None, fine_threshold=None):
    """Construct an eval-mode matcher with checkpoint-aware TCMCoarse settings."""
    from models import PairMatcher

    checkpoint = read_checkpoint(path)
    config = checkpoint_config(checkpoint)
    if coarse_threshold is not None:
        config["match_coarse"]["thr"] = coarse_threshold
    if fine_threshold is not None:
        config["fine"]["thr"] = fine_threshold
    matcher = PairMatcher(config)
    matcher.load_state_dict(matcher_state_dict(checkpoint), strict=True)
    return matcher.eval()
