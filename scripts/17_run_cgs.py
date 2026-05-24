"""Launcher for CodeGPTSensor's run.py — applies our torchvision shim, then defers
entirely to their unmodified script.

The dfc conda env has a broken `torchvision` (see [[dfc-env-quirks]]) which causes
`from transformers import RobertaModel` to fail with a circular-import error. We can't
patch the env (shared) and we won't patch their code (the project's rule). The minimum
intervention is to import `src._env` before transformers is imported, then exec their
`run.py` with the CLI args we were given.

Usage (drop-in for `python external/CodeGPTSensor/CodeGPTSensor/run.py …`):
  CUDA_VISIBLE_DEVICES=0 python scripts/17_run_cgs.py --do_train --model_name_or_path microsoft/unixcoder-base-nine \\
    --train_data_file data/hmcorp/python/train.jsonl --eval_data_file data/hmcorp/python/valid.jsonl \\
    --output_dir external/CodeGPTSensor/output/python_raw --num_train_epochs 20 \\
    --block_size 400 --train_batch_size 8 --eval_batch_size 16 --learning_rate 2e-5 \\
    --max_grad_norm 1.0 --seed 99 --contrast
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# 1) torchvision shim must be applied BEFORE any transformers import inside run.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import _env  # noqa: F401  applies the torchvision attribute shim

# 2) Put their CodeGPTSensor source dir on sys.path so `from model import Model` and
#    `from utils.early_stopping import EarlyStopping` resolve against their files.
CGS_DIR = ROOT / "external" / "CodeGPTSensor" / "CodeGPTSensor"
sys.path.insert(0, str(CGS_DIR))

# 3) Pretend we are their run.py — their argparse reads sys.argv[1:] which is already
#    whatever we were called with. Run with run_name='__main__' so their
#    `if __name__ == "__main__": main()` block fires.
sys.argv[0] = str(CGS_DIR / "run.py")
runpy.run_path(str(CGS_DIR / "run.py"), run_name="__main__")
