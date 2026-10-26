"""Homography-supervised datasets and training data loader."""

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset

from data.io import build_augmentor, read_multi_modality_gray


class MultiModality(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        img_resize=None,
        df=None,
        img_padding=False,
        augment_fn=None,
        **kwargs,
    ):
        """
        Read homography-supervised image pairs in level_* or BIT layouts.

        Args:
            root_dir (str): directory containing level_*/<modality>/<sample_id> folders.
            mode (str): training mode; augmentation is applied only for "train".
            img_resize (int, optional): the longer edge of resized images. None for no resize.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = []
        self.build_dataset()

        # parameters for image resizing and padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        self.augment_fn = augment_fn if mode == "train" else None
        self.coarse_scale = kwargs.get("coarse_scale", 0.125)

    def build_dataset(self):
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        # Accept either the common root (containing level_30/level_60) or one
        # selected level directory, such as .../level_30.
        root_name = os.path.basename(os.path.normpath(self.root_dir))
        if root_name.startswith("level_"):
            level_paths = [self.root_dir]
        else:
            level_paths = [
                os.path.join(self.root_dir, name)
                for name in sorted(os.listdir(self.root_dir))
                if name.startswith("level_") and os.path.isdir(os.path.join(self.root_dir, name))
            ]

        for level_path in level_paths:
            for modality in sorted(os.listdir(level_path)):
                modality_path = os.path.join(level_path, modality)
                if not os.path.isdir(modality_path):
                    continue
                for sample_id in sorted(os.listdir(modality_path)):
                    sample_path = os.path.join(modality_path, sample_id)
                    if not os.path.isdir(sample_path):
                        continue
                    image0_path = os.path.join(sample_path, "00.png")
                    image1_path = os.path.join(sample_path, "01.png")
                    matrix_path = os.path.join(sample_path, "01.txt")
                    if all(
                        os.path.isfile(path) for path in (image0_path, image1_path, matrix_path)
                    ):
                        self.dataset.append(
                            {
                                "image0_path": image0_path,
                                "image1_path": image1_path,
                                "affine_matrix_path": matrix_path,
                            }
                        )

        # BIT multi-modal layout:
        # root/<scene>/<scene_pair>/<folder>/<reference and warped files>.
        # Keep the repository's original pairing rule for compatibility.
        if not self.dataset:
            for scene in sorted(os.listdir(self.root_dir)):
                scene_path = os.path.join(self.root_dir, scene)
                if not os.path.isdir(scene_path):
                    continue
                for scene_pair in sorted(os.listdir(scene_path)):
                    scene_pair_path = os.path.join(scene_path, scene_pair)
                    if not os.path.isdir(scene_pair_path):
                        continue
                    for folder in sorted(os.listdir(scene_pair_path)):
                        folder_path = os.path.join(scene_pair_path, folder)
                        if not os.path.isdir(folder_path):
                            continue
                        files = sorted(os.listdir(folder_path))
                        for i in range(3, len(files)):
                            if i % 2 == 1 and i + 1 < len(files):
                                paths = (
                                    os.path.join(folder_path, files[0]),
                                    os.path.join(folder_path, files[i]),
                                    os.path.join(folder_path, files[i + 1]),
                                )
                                if all(os.path.isfile(path) for path in paths):
                                    self.dataset.append(
                                        {
                                            "image0_path": paths[0],
                                            "image1_path": paths[1],
                                            "affine_matrix_path": paths[2],
                                        }
                                    )

        if not self.dataset:
            raise RuntimeError(
                "No training pairs found in either supported layout: "
                "level_*/<modality>/<id> or the BIT layout. "
                f"Dataset root: {self.root_dir}"
            )
        np.random.RandomState(42).shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image0, mask0, scale0 = read_multi_modality_gray(
            item["image0_path"],
            self.img_resize,
            self.df,
            self.img_padding,
            augment_fn=self.augment_fn,
            random_gamma=self.mode == "train",
        )
        image1, mask1, scale1 = read_multi_modality_gray(
            item["image1_path"],
            self.img_resize,
            self.df,
            self.img_padding,
            augment_fn=self.augment_fn,
            random_gamma=self.mode == "train",
        )

        # read and compute relative poses
        T_0to1 = torch.from_numpy(
            np.loadtxt(item["affine_matrix_path"]).astype(np.float32)
        )  # (3, 3)
        T_1to0 = T_0to1.inverse()

        data = {
            "image0": image0,  # (1, h, w)
            "image1": image1,
            "T_0to1": T_0to1,  # (3, 3)
            "T_1to0": T_1to0,
            "scale0": scale0,  # [scale_w, scale_h]
            "scale1": scale1,
            "dataset_name": "Multi-Modality",
            "pair_id": idx,
        }

        # Downsample padding masks to the coarse feature resolution.
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(
                    torch.stack([mask0, mask1], dim=0)[None].float(),
                    scale_factor=self.coarse_scale,
                    mode="nearest",
                    recompute_scale_factor=False,
                )[0].bool()
            data.update({"mask0": ts_mask_0, "mask1": ts_mask_1})

        return data


class MultiSceneDataModule(pl.LightningDataModule):
    """Load homography-supervised image pairs for training."""

    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        if not self.train_data_root:
            raise ValueError("DATASET.TRAIN_DATA_ROOT must point to the training set.")

        # 2. dataset config
        # general options
        self.augment_fn = build_augmentor(
            config.DATASET.AUGMENTATION_TYPE
        )  # None, options: [None, 'dark', 'mobile']

        # Image sizing options
        self.mtmd_img_resize = config.DATASET.IMAGE_SIZE
        self.mtmd_img_pad = config.DATASET.PAD_IMAGES
        self.mtmd_df = config.DATASET.DIVISIBILITY
        self.coarse_scale = 1 / config.PAIR_MATCHER.RESOLUTION[0]

        # 3. loader parameters
        self.train_loader_params = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": getattr(args, "pin_memory", True),
        }

    def setup(self, stage=None):
        """
        Setup train dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase.
        """

        if stage not in (None, "fit"):
            raise ValueError("This data module supports training (fit) only.")

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0

        if stage in (None, "fit"):
            self.train_dataset = MultiModality(
                self.train_data_root,
                mode="train",
                img_resize=self.mtmd_img_resize,
                df=self.mtmd_df,
                img_padding=self.mtmd_img_pad,
                augment_fn=self.augment_fn,
                coarse_scale=self.coarse_scale,
            )
            logger.info(f"[rank:{self.rank}] Train Dataset loaded!")

    def train_dataloader(self):
        """
        Build training dataloader for MultiModality.
        """
        dataloader = DataLoader(self.train_dataset, **self.train_loader_params)
        return dataloader
