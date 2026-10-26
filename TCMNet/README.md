# TCMNet

TCMNet is a tri-modal image matching network based on **cycle consistency**. It uses a shared pairwise matcher to establish correspondences in three directions (1→2, 2→3, and 3→1), and improves matching results through cycle-based correction, completion, and filtering.

## 1. Installation

The verified environment is:

- Windows
- Python 3.9.25
- PyTorch 2.8.0
- CUDA 12.8

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Dataset Structure

The recommended training dataset structure is:

```text
<data-root>/
  level_30/
    <modality>/
      <sample-id>/
        00.png
        01.png
        01.txt
```

Here, `00.png` and `01.png` are paired images, and `01.txt` contains a 3×3 homography matrix mapping coordinates from the original `00.png` to the original `01.png`.

## 3. Training

Train the model using:

```bash
python train.py --config configs/train.yaml --data-config configs/data.yaml --data-root path/to/dataset
```

By default, training initializes from `weights/weights.ckpt` and jointly optimizes the backbone, TCM-Coarse, fine matching, and subpixel regression modules.

To train from scratch:

```bash
python train.py --config configs/train.yaml --data-config configs/data.yaml --data-root path/to/dataset --ckpt-path ''
```

## 4. Inference

Perform tri-modal cycle matching with three input images:

```bash
python infer_tcmnet.py --modality1 testimg/01/00.png --modality2 testimg/01/01.png --modality3 testimg/01/a01.png --weights weights/weights.ckpt --viz outputs/example.png
```

The model uses GPU by default. Input images are converted to grayscale and resized to 512×512.

Optional arguments:

- `--cpu`: Run inference on CPU.
- `--no-viz`: Disable visualization.
- `--no-filter`: Disable geometric filtering.
- `--weights`: Specify a model checkpoint.

## 5. Evaluation

Evaluate matching results using a ground-truth homography:

```bash
python infer_tcmnet.py --modality1 testimg/01/00.png --modality2 testimg/01/01.png --modality3 testimg/01/a01.png --gt testimg/01/01.txt --gt-direction rs --no-viz
```

The evaluation reports:

- `matches`: Number of matched point pairs.
- `inliers`: Number of inlier matches.
- `ratio`: Inlier ratio.
- `reproj`: Mean inlier reprojection error.

Results are reported for both the original pairwise matches (`baseline`) and the cycle-enhanced matches (`enhanced`).

When ground-truth homography is unavailable, MAGSAC is used to estimate geometric consistency from the predicted matches.