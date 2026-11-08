"""Step 1 of the GPT-4.1 caption pipeline (Section 4.2): Molmo-7B draft captions.

Molmo-7B is prompted with the observation metadata and the image. Its drafts
hallucinate visual traits and metadata, so they are only used as input to the
GPT-4.1 correction step (captions/gpt_batch.py), never for training directly.

    python -m captions.molmo_caption --data-root DATA --out data/captions/molmo_train.csv
"""

import argparse

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from captions.prompts import metadata_prompt
from fungiclef.data import load_split

MODEL = "allenai/Molmo-7B-D-0924"

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--split", default="train")
parser.add_argument("--out", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(42)

    processor = AutoProcessor.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, device_map="auto"
    )

    rows = []
    for _, row in tqdm(load_split(args.data_root, args.split).iterrows()):
        inputs = processor.process(
            images=[Image.open(row["path"]).convert("RGB")], text=metadata_prompt(row)
        )
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        out = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
        text = processor.tokenizer.decode(
            out[0, inputs["input_ids"].size(1):], skip_special_tokens=True
        )
        rows.append({"filename": row["filename"], "caption": text.replace("\n", "").strip()})

    pd.DataFrame(rows).to_csv(args.out, index=False)
