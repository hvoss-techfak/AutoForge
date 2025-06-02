#!/usr/bin/env python3
"""
Grid-runner for autoforge.py

Usage example (inline defaults):
    python run_autoforge_grid.py
"""

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from time import strftime

# ---------- editable section ---------- #
# Any CLI flag accepted by autoforge.py can be placed here.
DEFAULT_ARGS: dict[str, object] = {
    "--csv_file": "bambulab.csv",
    "--iterations": 2000,
    "--stl_output_size": 50,
    "--visualize": False,
}

SWEEP_PARAM = "--offset_lr_strength"  # param to overwrite per run
SWEEP_VALUES = [0.01, 0.1, 1, 10]

IMAGES_DIR = Path("images/test_images")
BASE_OUTPUT_DIR = Path("output_grid")  # all run folders are created inside here
MAX_WORKERS = 2  # parallel jobs
# ---------- end editable section ------ #


def make_cmd(image_path: Path, param_value, run_dir: Path) -> list[str]:
    """Assemble the command line for one run."""
    cmd: list[str] = [
        sys.executable,
        "autoforge.py",
        "--input_image",
        str(image_path),
        "--output_folder",
        str(run_dir),
    ]

    # default args
    for flag, val in DEFAULT_ARGS.items():
        if isinstance(val, bool):
            if val:  # store_true flag
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    # swept parameter
    cmd.extend([SWEEP_PARAM, str(param_value)])
    return cmd


def run_single(image_path: Path, param_value) -> dict:
    """Worker: launch subprocess, parse loss, return result dict."""
    run_dir = BASE_OUTPUT_DIR / (
        f"{image_path.stem}_{SWEEP_PARAM.lstrip('-')}={param_value}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = make_cmd(image_path, param_value, run_dir)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    loss_file = run_dir / "final_loss.txt"
    loss = None
    if loss_file.exists():
        try:
            loss = float(loss_file.read_text().strip())
        except ValueError:
            pass

    return {
        "image": str(image_path),
        "param_value": param_value,
        "loss": loss,
        "output_folder": str(run_dir),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    values = SWEEP_VALUES

    images = [
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # run grid
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single, img, val): (img, val)
            for img, val in product(images, values)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(
                f"✓ finished {Path(res['image']).name} "
                f"{SWEEP_PARAM}={res['param_value']} loss={res['loss']}"
            )

    # write summary
    ts = strftime("%Y%m%d_%H%M%S")
    out_path = BASE_OUTPUT_DIR / f"out_dict_{ts}.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
