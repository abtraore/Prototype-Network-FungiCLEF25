#!/bin/bash
# CLIP finetuning on image-caption pairs (Section 4.2, Table 2).
#   bash clip_finetune/finetune.sh DATA gpt41     # GPT-4.1 captions
#   bash clip_finetune/finetune.sh DATA qwen2vl   # Qwen2-VL-2B captions
# ViT-H-14-378-quickgelu (DFN-5B), batch 5, lr 1e-5, wd 0.01, 10 epochs, AMP,
# cosine schedule with warmup. Evaluate the last checkpoint with
#   python -m baseline.prototype_averaging --data-root DATA --model vit-h-14-dfn5b \
#       --checkpoint logs/clip-<captions>/checkpoints/epoch_latest.pt
set -e
DATA=$1
CAPTIONS=${2:-gpt41}

python -m clip_finetune.make_csv --data-root "$DATA" --captions "$CAPTIONS"

python -m clip_finetune.train \
    --train-data "data/openclip/${CAPTIONS}_train.csv" \
    --val-data "data/openclip/${CAPTIONS}_val.csv" \
    --csv-img-key filename \
    --csv-caption-key caption \
    --csv-separator "," \
    --model ViT-H-14-378-quickgelu \
    --pretrained dfn5b \
    --batch-size 5 \
    --lr 1e-5 \
    --wd 0.01 \
    --warmup 3200 \
    --epochs 10 \
    --lr-scheduler cosine \
    --precision amp \
    --workers 8 \
    --zeroshot-frequency 0 \
    --save-most-recent \
    --save-frequency 9999 \
    --report-to tensorboard \
    --logs logs \
    --name "clip-${CAPTIONS}"
