"""Qwen2-VL-2B supervised finetuning on the GPT-4.1 captions (Section 4.2).

4-bit quantised base model + LoRA (rank 8, alpha 16, dropout 0.05), 5 epochs,
per-device batch 4, gradient accumulation 8, lr 2e-4, max grad norm 0.3,
warmup ratio 0.03, cosine schedule, bf16. Run with captions/qwen_sft.sh.
"""

from dataclasses import dataclass

import pandas as pd
import torch
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser

from captions.prompts import SFT_PROMPT, SFT_USER
from fungiclef.data import image_dir

QWEN_IMAGE_TOKENS = [151652, 151653, 151655]  # vision start/end/pad, masked from the loss


@dataclass
class DataArguments:
    data_root: str = ""
    train_captions: str = "data/captions/gpt41_train.csv"
    val_captions: str = "data/captions/gpt41_val.csv"


def resize_max_224(image):
    """Downscale so the longest side is 224 px, keeping the aspect ratio."""
    w, h = image.size
    if max(w, h) <= 224:
        return image
    s = 224 / max(w, h)
    return image.resize((int(w * s), int(h * s)), Image.LANCZOS)


def conversation(image_path, caption=None):
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SFT_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SFT_USER}]},
    ]
    if caption is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
    return {"messages": msgs, "image": image_path}


def load_dataset(data_root, csv, split):
    root = image_dir(data_root, split)
    df = pd.read_csv(csv)
    return [conversation(str(root / f), c) for f, c in zip(df["filename"], df["caption"])]


if __name__ == "__main__":
    parser = TrlParser((DataArguments, ScriptArguments, SFTConfig, ModelConfig))
    data_args, _, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.label_names = ["labels"]

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )

    def collate(examples):
        texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
        images = [resize_max_224(Image.open(e["image"]).convert("RGB")) for e in examples]
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tok in QWEN_IMAGE_TOKENS:
            labels[labels == tok] = -100
        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=load_dataset(data_args.data_root, data_args.train_captions, "train"),
        eval_dataset=load_dataset(data_args.data_root, data_args.val_captions, "val"),
        data_collator=collate,
        processing_class=processor.tokenizer,
        peft_config=LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
