"""Section 4.1 (Table 1) and evaluation of Section 4.2 (Table 2).

Training-free prototype averaging with a pretrained OpenCLIP image encoder.
Pass --checkpoint to evaluate a CLIP model finetuned with clip_finetune/.

    python -m baseline.prototype_averaging --model bioclip --support trainval
    python -m baseline.prototype_averaging --model vit-h-14-dfn5b --support trainval
    python -m baseline.prototype_averaging --model vit-b-16-dfn2b --support trainval
    python -m baseline.prototype_averaging --model vit-h-14-dfn5b \
        --checkpoint logs/<run>/checkpoints/epoch_10.pt
"""

import argparse

import open_clip
import torch
from PIL import Image

from fungiclef.data import load_split, load_support
from fungiclef.prototypes import build_prototypes, predict_top5

MODELS = {
    # name: (open_clip architecture, pretrained tag)
    "bioclip": ("hf-hub:imageomics/bioclip", None),  # ViT-B-16-224, TreeOfLife-10M
    "vit-h-14-dfn5b": ("ViT-H-14-378-quickgelu", "dfn5b"),
    "vit-b-16-dfn2b": ("ViT-B-16-quickgelu", "dfn2b"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--model", choices=MODELS, default="bioclip")
parser.add_argument("--support", choices=["train", "trainval"], default="trainval")
parser.add_argument("--checkpoint", help="finetuned open_clip checkpoint (Table 2)")
parser.add_argument("--device", default="cuda")
parser.add_argument("--out", default="submission.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    arch, tag = MODELS[args.model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=None if args.checkpoint else tag
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model = model.to(args.device).eval()

    def transform(path):
        return preprocess(Image.open(path).convert("RGB"))

    support = load_support(args.data_root, args.support)
    ids, protos = build_prototypes(model.encode_image, transform, support, args.device)
    sub = predict_top5(
        model.encode_image, transform, load_split(args.data_root, "test"),
        ids, protos, args.device,
    )
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
