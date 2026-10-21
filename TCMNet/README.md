# TCMNet

TCMNet uses a shared pairwise matcher for three images in the 1→2, 2→3, and 3→1 directions. During inference, it uses the tri-modal cycle to correct, complete, and filter matches. `train.py` trains the pairwise matcher's backbone, TCM-Coarse, local fine matching, and subpixel regression. Cycle processing is not currently included in the training loss.

## Installation

The verified environment is Windows with Python 3.9.25, PyTorch 2.8.0, and CUDA 12.8. For a different CUDA version or a CPU-only system, install compatible PyTorch and torchvision builds first, then install the remaining dependencies. See `requirements.txt` for the full dependency list and versions.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

Inference uses an available CUDA device by default; pass `--cpu` to use the CPU. Input images are converted to grayscale and resized to 512×512. Model weights are loaded from `weights/weights.ckpt` by default; use `--weights` to specify another checkpoint.

## Data format

### Three-image inference input

`infer_tcmnet.py` takes three images at a time; their filenames are unrestricted. The order of `--modality1`, `--modality2`, and `--modality3` determines the internal directions: `rs` is 1→2, `si` is 2→3, and `ir` is 3→1. These direction names are internal identifiers and do not restrict the actual modalities. For example, the inputs may be RGB, SAR, and IR in that order.

```powershell
python infer_tcmnet.py --modality1 testimg/01/00.png --modality2 testimg/01/01.png --modality3 testimg/01/a01.png --weights weights/weights.ckpt --viz outputs/example.png
```

### Pairwise training input

The recommended training data layout is shown below. `level_30` is only an example: `--data-root` may point to a directory containing several `level_*` subdirectories or directly to one `level_*` directory.

```text
<data-root>/
  level_30/
    <modality>/
      <sample-id>/
        00.png
        01.png
        01.txt
```

Each `01.txt` is a plain-text 3×3 homography that maps pixel coordinates from the **original** `00.png` to the original `01.png`: `[x', y', 1]ᵀ ∝ H [x, y, 1]ᵀ`. The matrix must be invertible; the data loader computes its inverse. During training, images are resized so their longer side is 512, then padded according to the configuration. The data loader also retains the coordinate scale factors. The older BIT directory layout is supported, but new datasets should use the `level_*` layout above.

```powershell
python train.py --config configs/train.yaml --data-config configs/data.yaml --data-root path/to/dataset
```

Training initializes from `weights/weights.ckpt` by default. To train the `full` scope from scratch, pass `--ckpt-path ''`. The default `full` scope jointly updates all parts of the pairwise matcher; `--train-scope tcm-coarse-only` updates only TCM-Coarse.

## Evaluation

The inference command reports `matches`, `inliers`, `ratio`, and `reproj` for both the original pairwise matches (`baseline`) and the cycle-processed matches (`enhanced`). The default `--ransac-thresh` is 3, measured in pixels of the resized 512×512 images.

**With a ground-truth homography**, pass a 3×3 `.txt` or `.npy` matrix via `--gt` and specify its direction (`rs`, `si`, or `ir`) with `--gt-direction`. The matrix must be defined in the **original image coordinates**. The command transforms it to 512×512 coordinates, then computes each match's reprojection error, the number of inliers within the threshold, and the mean error over those inliers. If ground truth is provided for only one direction, the other two are evaluated using fitted geometry as described below.

```powershell
python infer_tcmnet.py --modality1 testimg/01/00.png --modality2 testimg/01/01.png --modality3 testimg/01/a01.png --gt testimg/01/01.txt --gt-direction rs --no-viz
```

If the same original 3×3 matrix truly maps modality 1 to both modalities 2 and 3, add `--gt-shared-modality1` to derive ground truth for 2→3 and 3→1. Do not use this option when the two target images have different transformations.

**Without ground truth**, the command fits a MAGSAC homography to the output matches and reports the fitted inlier count, inlier ratio, and mean inlier reprojection error. These metrics measure internal geometric consistency, not ground-truth matching accuracy. Output geometry filtering is enabled by default; pass `--no-filter` to disable it or `--filter` to enable it explicitly. This filtering is separate from the MAGSAC fit used to report metrics. The command evaluates one set of three images at a time; dataset-wide summary statistics are not implemented.
