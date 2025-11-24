"""Accelerate launcher wrapper for existing `train.py`.

This script is a minimal, safe wrapper that invokes the `accelerate launch`
command with the repository `accelerate_fsdp_config.yaml` and forwards all
arguments to the original training entrypoint `lerobot/src/lerobot/scripts/train.py`.

Usage examples:

# Interactive: configure accelerate once
accelerate config

# Launch using the included config file (adjust num_processes in the YAML)
python lerobot/src/lerobot/scripts/train_accelerate.py -- --epochs 1 --batch_size 2

# If you want to pass additional args to `train.py`, place them after the first `--`.

Notes:
- This wrapper uses the `accelerate` CLI if available in PATH. If not, it
  falls back to running accelerate with the Python module entrypoint.
- The wrapper does not change training logic; it simply wraps the launch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = REPO_ROOT / "accelerate_fsdp_config.yaml"
TRAIN_SCRIPT = REPO_ROOT / "src" / "lerobot" / "scripts" / "train.py"
# /home/bastian/ws25_ms_silder_vlam/lerobot/src/lerobot/scripts/train.py

def build_accelerate_cmd(argv: list[str]) -> list[str]:
    """Build the accelerate launch command, using the accel binary if available,
    otherwise falling back to `python -m accelerate.commands.launch`.
    The wrapper will forward any extra args to the training script after "--".
    """
    extra = argv

    accel_bin = shutil.which("accelerate")
    if accel_bin:
        cmd = [accel_bin, "launch", "--config_file", str(CONFIG_FILE), str(TRAIN_SCRIPT), "--"]
    else:
        # fallback to running accelerate as a module. The module path can vary
        # between accelerate versions; this is a common entrypoint.
        cmd = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", str(CONFIG_FILE), str(TRAIN_SCRIPT), "--"]

    cmd.extend(extra)
    return cmd


def main():
    if not CONFIG_FILE.exists():
        print(f"ERROR: accelerate config not found at {CONFIG_FILE}")
        print("Please create or adapt the accelerate_fsdp_config.yaml file in the repo root.")
        sys.exit(2)

    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: training entrypoint not found at {TRAIN_SCRIPT}")
        sys.exit(2)

    # find the separator ' -- ' if present. This script accepts args that will
    # be forwarded directly to train.py. For convenience you can pass args to
    # this wrapper which will be forwarded.
    # Example: python train_accelerate.py -- --epochs 10 --batch_size 8
    args = sys.argv[1:]

    # Command to run
    cmd = build_accelerate_cmd(args)

    print("Launching training with accelerate:")
    print(" ".join(cmd))

    # Run the command and forward stdout/stderr. Return same exit code.
    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        print("Failed to start accelerate launch. Is `accelerate` installed and available in PATH?")
        print(e)
        sys.exit(3)


if __name__ == "__main__":
    main()
