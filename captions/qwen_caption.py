"""Caption every train/val image with the finetuned Qwen2-VL-2B (Section 4.2).

    python -m captions.qwen_caption --data-root DATA --adapter logs/qwen2vl-fungi --split train

Writes data/captions/qwen2vl_{split}.csv (already released in this repo).
"""

import argparse

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from captions.qwen_sft import conversation, resize_max_224
from fungiclef.data import load_split

BASE = "Qwen/Qwen2-VL-2B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--split", choices=["train", "val"], default="train")
parser.add_argument("--max-new-tokens", type=int, default=128)

if __name__ == "__main__":
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(BASE, use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    rows = []
    for _, row in tqdm(load_split(args.data_root, args.split).iterrows()):
        conv = conversation(row["path"])
        prompt = processor.apply_chat_template(conv["messages"], tokenize=False, add_generation_prompt=True)
        image = resize_max_224(Image.open(row["path"]).convert("RGB"))
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        caption = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"filename": row["filename"], "caption": caption.strip()})

    pd.DataFrame(rows).to_csv(f"data/captions/qwen2vl_{args.split}.csv", index=False)
