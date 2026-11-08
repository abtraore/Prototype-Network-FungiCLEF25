"""Episodic training of the prototypical network (Section 4.3, Table 3).

Table 3 rows:

    # BioCLIP, train/val, K=75 S=3 Q=1, 750 episodes (best: 0.64159 PB / 0.58473 PR)
    python -m protonet.train --encoder bioclip --split trainval --k 75 --episodes 750
    # BioCLIP, train, K=70 S=3 Q=1, 500 episodes
    python -m protonet.train --encoder bioclip --split train --k 70 --episodes 500
    # ViT-B-16-SigLIP-512, train, K=15 S=3 Q=1, 750 episodes
    python -m protonet.train --encoder siglip-512 --split train --k 15 --episodes 750

Optimisation: AdamW, lr 1e-5, cross-entropy over negative squared L2 logits,
Stochastic Weight Averaging (SWA lr 5e-5) over the last 100 episodes.
One "episode" is one optimiser step; its gradient is accumulated over
--accum sampled episodes (32 in all paper runs) to fit in GPU memory.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import tqdm

from fungiclef.data import load_support
from protonet.episodes import EpisodeSampler, make_transform
from protonet.model import ENCODERS, ProtoNet

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--encoder", choices=ENCODERS, default="bioclip")
parser.add_argument("--split", choices=["train", "trainval"], default="trainval")
parser.add_argument("--k", type=int, default=75, help="classes (ways) per episode")
parser.add_argument("--s", type=int, default=3, help="support images (shots) per class")
parser.add_argument("--episodes", type=int, default=750)
parser.add_argument("--swa-episodes", type=int, default=100)
parser.add_argument("--accum", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--swa-lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--out", default="protonet.pt")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    size = ENCODERS[args.encoder][2]
    sampler = EpisodeSampler(
        load_support(args.data_root, args.split), make_transform(size), args.k, args.s
    )

    model = ProtoNet(args.encoder).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    swa_start = args.episodes - args.swa_episodes

    for episode in tqdm(range(args.episodes)):
        for _ in range(args.accum):
            support, query, target = sampler.sample()
            logits = model(support.to(args.device), query.to(args.device))
            loss = F.cross_entropy(logits, target.to(args.device)) / args.accum
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if episode >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    torch.save(swa_model.module.state_dict(), args.out)
    print(f"wrote {args.out}")
