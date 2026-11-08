"""Turn data/captions/<name>_{train,val}.csv into open_clip CSVs with absolute image paths."""

import argparse
from pathlib import Path

import pandas as pd

from fungiclef.data import image_dir

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--captions", default="gpt41")

if __name__ == "__main__":
    args = parser.parse_args()
    Path("data/openclip").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        df = pd.read_csv(f"data/captions/{args.captions}_{split}.csv")
        root = image_dir(args.data_root, split)
        df["filename"] = [str(root / f) for f in df["filename"]]
        df[["filename", "caption"]].to_csv(f"data/openclip/{args.captions}_{split}.csv", index=False)
