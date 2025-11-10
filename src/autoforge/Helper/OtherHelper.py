import os
import sys
import time
from typing import Any

import numpy as np
import torch


def set_seed(args) -> Any:
    random_seed = args.random_seed
    if random_seed == 0:
        random_seed = int(time.time() * 1000) % 1000000
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    return random_seed


def perform_basic_check(args):
    # Basic checks
    if not (args.background_height / args.layer_height).is_integer():
        print(
            "Error: Background height must be a multiple of layer height.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.csv_file != "" and not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.json_file != "" and not os.path.exists(args.json_file):
        print(f"Error: Json file '{args.json_file}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.priority_mask != "" and not os.path.exists(args.priority_mask):
        print(
            f"Error: priority mask file '{args.priority_mask}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_device(args) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device
