"""Inference with a trained prototypical network (Section 4.3).

Class prototypes are built from all train and validation images, then every
test observation is ranked against them by cosine similarity (top-5).

    python -m protonet.predict --encoder bioclip --weights protonet.pt
"""

import argparse

import torch

from fungiclef.data import load_split, load_support
from fungiclef.prototypes import build_prototypes, predict_top5
from protonet.episodes import make_transform
from protonet.model import ENCODERS, ProtoNet

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--encoder", choices=ENCODERS, default="bioclip")
parser.add_argument("--weights", required=True)
parser.add_argument("--support", choices=["train", "trainval"], default="trainval")
parser.add_argument("--device", default="cuda")
parser.add_argument("--out", default="submission.csv")
parser.add_argument("--save-prototypes", help="optional .pt file, used by analysis/umap.py")

if __name__ == "__main__":
    args = parser.parse_args()
    model = ProtoNet(args.encoder, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = model.to(args.device).eval()
    transform = make_transform(ENCODERS[args.encoder][2])

    ids, protos = build_prototypes(
        model.encode, transform, load_support(args.data_root, args.support), args.device
    )
    if args.save_prototypes:
        torch.save({"category_id": ids, "prototypes": protos.cpu()}, args.save_prototypes)
    sub = predict_top5(
        model.encode, transform, load_split(args.data_root, "test"), ids, protos, args.device
    )
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
