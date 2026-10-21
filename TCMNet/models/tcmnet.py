"""TCMNet: three shared coarse branches followed by integrated TCM-Fine."""

import torch
from torch import nn

from configs.default import PAIR_SPECS
from cycle.tcm_fine import TCMFine


class TCMNet(nn.Module):
    """Run modality 1→2, 2→3 and 3→1 matching with one shared matcher."""

    def __init__(
        self,
        pair_matcher,
        *,
        verify_derived=True,
        refine_cycles=True,
        verifier_options=None,
        cycle_refiner_options=None,
        **fine_options,
    ):
        super().__init__()
        self.pair_matcher = pair_matcher
        self.tcm_fine = TCMFine(**fine_options)
        self.verify_derived = verify_derived
        self.refine_cycles = refine_cycles
        self.verifier_options = verifier_options or {}
        self.cycle_refiner_options = cycle_refiner_options or {}

    @torch.no_grad()
    def forward(self, modality1, modality2, modality3):
        images = {
            "modality1": modality1,
            "modality2": modality2,
            "modality3": modality3,
        }
        for name, image in images.items():
            if image.ndim != 4 or image.shape[:2] != (1, 1):
                raise ValueError(f"{name} must have shape [1, 1, H, W].")
            if not torch.is_floating_point(image):
                raise TypeError(f"{name} must be a floating-point tensor.")
        states = {}
        for pair, (source, target) in PAIR_SPECS.items():
            states[pair] = self.pair_matcher.forward_coarse(
                {"image0": images[source], "image1": images[target]}
            )
        return self.tcm_fine.from_coarse_states(
            self.pair_matcher,
            states,
            verify_derived=self.verify_derived,
            refine_cycles=self.refine_cycles,
            verifier_options=self.verifier_options,
            cycle_refiner_options=self.cycle_refiner_options,
        )
