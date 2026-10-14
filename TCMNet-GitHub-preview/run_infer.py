"""Public entry point for the TCMNet inference preview."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'runtime'))
if '--weights' not in sys.argv:
    sys.argv.extend(['--weights', str(root / 'weights' / 'weights.ckpt')])
from infer_tcmnet import main

if __name__ == '__main__':
    main()
