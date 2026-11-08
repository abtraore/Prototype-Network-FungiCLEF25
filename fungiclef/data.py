"""FungiTastic-FewShot paths, metadata loading and image reading.

Expected layout under --data-root (as distributed for FungiCLEF 2025):

    images/FungiTastic-FewShot/{train,val,test}/fullsize/*.JPG
    metadata/FungiTastic-FewShot/FungiTastic-FewShot-{Train,Val,Test}.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# OpenAI CLIP normalisation, used for every model in the paper.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def image_dir(data_root, split):
    return Path(data_root) / "images" / "FungiTastic-FewShot" / split / "fullsize"


def load_split(data_root, split):
    """Metadata of one split, with an absolute `path` column added."""
    meta = Path(data_root) / "metadata" / "FungiTastic-FewShot"
    df = pd.read_csv(meta / f"FungiTastic-FewShot-{split.capitalize()}.csv")
    df["path"] = [str(image_dir(data_root, split) / f) for f in df["filename"]]
    return df


def load_support(data_root, support):
    """Labelled images used to build prototypes: `train` or `trainval`."""
    dfs = [load_split(data_root, "train")]
    if support == "trainval":
        dfs.append(load_split(data_root, "val"))
    return pd.concat(dfs, ignore_index=True)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))
