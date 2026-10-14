# TCMNet inference preview

This is a **pre-release inference preview**, not the final research release. The learned matcher and tri-modal refinement implementation are provided as Python 3.9 bytecode; core Python source and training code are not included. Bytecode can be reverse engineered and is not a security boundary.

## Platform and setup

Tested on Windows x64 with Python 3.9 and PyTorch 2.8. Install the PyTorch build appropriate for your CPU/CUDA setup, then run:

```powershell
python -m pip install -r requirements.txt
```

Download the separate pre-release ZIP and extract it, or put the provided checkpoint at `weights/weights.ckpt`. The checkpoint is not committed to Git because it is over GitHub's regular-file size limit.

## Inference

Each sample needs a reference image (`00.png`), a target image (`01.png`) and a third-modality image (for example, `se01.png`). Pass the paths explicitly; images and results are not part of this repo.

```powershell
python run_infer.py --rgb path/to/00.png --sar path/to/01.png --ir path/to/se01.png --filter --viz outputs/example.png
```

Run `python run_infer.py --help` for the full inference options. The three image labels name the matcher inputs; for other sample types, use the reference image, target image, and third image in that order.

## Scope

This preview supports inference and visualization only. It does not include private datasets, training scripts, or a source build. No open-source license has been granted with this preview.
