"""Image IO and photometric augmentation."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch


def imread_gray(path, augment_fn=None, to_thermal=False, random_gamma=False):
    """Read non-ASCII paths; augmentation is explicitly enabled by the dataset."""
    image = read_image(path, cv2.IMREAD_COLOR)
    if augment_fn is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(augment_fn(rgb), cv2.COLOR_RGB2BGR)
    if to_thermal:
        return RGBtoThermal().augment_pseudo_thermal(image)
    if random_gamma:
        choice = random.choices([0, 1, 2], [1 / 3, 2 / 9, 4 / 9])[0]
        gamma = (
            1.0
            if choice == 0
            else random.uniform(0.3, 0.6)
            if choice == 1
            else random.uniform(2, 4)
        )
        image = adjust_gamma(image, gamma=gamma)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), (
        f"{pad_size} < {max(inp.shape[-2:])}"
    )
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: inp.shape[0], : inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[1], : inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : inp.shape[1], : inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def read_multi_modality_gray(
    path, resize=None, df=None, padding=False, augment_fn=None, to_thermal=False, random_gamma=False
):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_gray(path, augment_fn, to_thermal=to_thermal, random_gamma=random_gamma)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale


class RGBtoThermal:
    def __init__(self):
        import albumentations as A

        self.blur = A.Blur(p=0.7, blur_limit=(3, 5))  # default: blur_limit=(2, 4)
        self.hsv = A.HueSaturationValue(
            p=0.9,
            val_shift_limit=(-30, +30),
            hue_shift_limit=(-90, +90),
            sat_shift_limit=(-30, +30),
        )

        # parameters for the cosine transform
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    def augment_pseudo_thermal(self, image):
        # HSV augmentation
        image = self.hsv(image=image)["image"]

        # Random blur
        image = self.blur(image=image)["image"]

        # Convert the image to the gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the image between (-0.5, 0.5)
        image = image / 255 - 0.5

        # Random phase and freq for the cosine transform
        phase = np.pi / 2 + np.random.randn(1) * self.theta_r
        w = self.w_0 + np.abs(np.random.randn(1)) * self.w_r

        # Cosine transform
        image = np.cos(image * w + phase)

        # Min-max normalization for the transformed image
        image = (image - image.min()) / max(float(image.max() - image.min()), 1e-12) * 255

        return image.astype(np.uint8)


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    return ((image / 255) ** inv_gamma * 255).astype("uint8")


class DarkAug(object):
    """
    Extreme dark augmentation aiming at Aachen Day-Night
    """

    def __init__(self) -> None:
        import albumentations as A

        self.augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(
                    p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            ],
            p=0.75,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class MobileAug(object):
    """
    Random augmentations aiming at images of mobile/handhold devices.
    """

    def __init__(self):
        import albumentations as A

        self.augmentor = A.Compose(
            [
                A.MotionBlur(p=0.25),
                A.ColorJitter(p=0.5),
                A.RandomRain(p=0.1),  # random occlusion
                A.RandomSunFlare(p=0.1),
                A.ImageCompression(p=0.25),
                A.ISONoise(p=0.25),
            ],
            p=1.0,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


def build_augmentor(method=None, **kwargs):
    if method == "dark":
        return DarkAug()
    elif method == "mobile":
        return MobileAug()
    elif method is None:
        return None
    else:
        raise ValueError(f"Invalid augmentation method: {method}")


def read_color(path: str) -> np.ndarray:
    return read_image(path, cv2.IMREAD_COLOR)


def read_image(path, flags=cv2.IMREAD_GRAYSCALE):
    """Decode a local image without losing non-ASCII filesystem paths."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: str, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Could not encode output image: {output}")
    encoded.tofile(str(output))
