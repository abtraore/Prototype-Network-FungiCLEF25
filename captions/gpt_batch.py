"""Steps 2 and 3 of the GPT-4.1 caption pipeline (Section 4.2), via the OpenAI Batch API.

    correct   : train split, GPT-4.1 rewrites each Molmo-7B draft using the image
                and metadata, discarding inaccurate content (structured output);
                val split, GPT-4.1 writes the caption from image + metadata.
    summarize : GPT-4.1 shortens each caption to <=70 tokens, appearance only,
                so it fits CLIP's 77-token text encoder.

Each stage is: build -> submit -> (wait) -> collect.

    python -m captions.gpt_batch build --stage correct --split train \
        --data-root DATA --drafts data/captions/molmo_train.csv
    python -m captions.gpt_batch submit --stage correct --split train
    python -m captions.gpt_batch collect --stage correct --split train   # -> long_train.csv
    python -m captions.gpt_batch build --stage summarize --split train \
        --data-root DATA --drafts data/captions/long_train.csv
    python -m captions.gpt_batch submit --stage summarize --split train
    python -m captions.gpt_batch collect --stage summarize --split train # -> gpt41_train.csv

Requires OPENAI_API_KEY. The resulting captions are already released in
data/captions/gpt41_{train,val}.csv.
"""

import argparse
import base64
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

from captions.prompts import (
    CORRECT_PROMPT, DIRECT_PROMPT, GPT_MODEL, SUMMARIZE_SYSTEM, SUMMARIZE_USER,
    metadata_prompt,
)
from fungiclef.data import load_split

CHUNK = 50  # requests per batch file

CORRECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "object",
            "properties": {
                "inaccuracies": {
                    "type": "object",
                    "properties": {
                        "visual_appearance": {
                            "type": "object",
                            "properties": {
                                "error_type": {"type": "string", "enum": ["missing", "inaccurate"]},
                                "description": {"type": "string"},
                            },
                            "required": ["error_type", "description"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["visual_appearance"],
                    "additionalProperties": {"type": "string"},
                    "description": "Field-specific inaccuracies",
                },
                "missing_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Omitted template fields",
                },
            },
            "required": ["inaccuracies", "missing_fields"],
            "additionalProperties": False,
        },
        "corrected_caption": {
            "type": "string",
            "description": (
                "Flowing narrative that integrates all data from the real prompt, "
                "and any valid visual characteristics detected in the image, "
                "following taxonomic conventions and using precise biological language."
            ),
        },
        "species": {
            "type": "string",
            "description": "Full scientific binomial with authority (e.g., Xylaria polymorpha (Pers.) Berk. & Broome)",
        },
    },
    "required": ["assessment", "corrected_caption", "species"],
    "additionalProperties": False,
}

DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_caption": {
            "type": "string",
            "description": (
                "Comprehensive, precise narrative caption conforming strictly to the prompt: "
                "fully integrating all field values verbatim from the real prompt (genus, scientificName, species, "
                "specificEpithet, kingdom, phylum, class, order, family, substrate, poisonous/non-poisonous) and "
                "describing all visible morphological/ecological image features. "
                "Begins by stating close-up/habitat/indeterminate framing."
            ),
        },
        "species": {
            "type": "string",
            "description": "Full scientific binomial with authority, copied verbatim from real prompt.",
        },
    },
    "required": ["corrected_caption", "species"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "A concise, appearance-focused description of the fungus in ≤70 tokens. No taxonomy or non-visual info.",
        },
        "species": {
            "type": "string",
            "description": "Full scientific binomial with authority (e.g., Xylaria polymorpha (Pers.) Berk. & Broome)",
        },
    },
    "required": ["caption", "species"],
    "additionalProperties": False,
}


def image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}


def text(t):
    return {"type": "input_text", "text": t}


def request(custom_id, system, user, name, schema):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": GPT_MODEL,
            "temperature": 0,
            "input": [
                {"role": "system", "content": [text(system)]},
                {"role": "user", "content": user},
            ],
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        },
    }


def build(args, work):
    meta = load_split(args.data_root, args.split).set_index("filename")
    drafts = pd.read_csv(args.drafts) if args.drafts else None
    reqs = []
    if args.stage == "correct" and args.split == "train":
        for _, d in drafts.iterrows():
            row = meta.loc[d["filename"]]
            user = [text(f"Real prompt: {metadata_prompt(row)}"), text(f"Caption: {d['caption']}"),
                    image_part(row["path"])]
            reqs.append(request(d["filename"], CORRECT_PROMPT, user,
                                "fungus_caption_assessment", CORRECTED_SCHEMA))
    elif args.stage == "correct":
        for filename, row in meta.iterrows():
            user = [text(f"Real prompt: {metadata_prompt(row)}"), image_part(row["path"])]
            reqs.append(request(filename, DIRECT_PROMPT, user, "fungus_direct_caption", DIRECT_SCHEMA))
    else:
        for _, d in drafts.iterrows():
            user = [text(SUMMARIZE_USER), image_part(meta.loc[d["filename"], "path"]), text(d["caption"])]
            reqs.append(request(d["filename"], SUMMARIZE_SYSTEM, user,
                                "fungus_caption_result", SUMMARY_SCHEMA))

    work.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(reqs), CHUNK):
        with open(work / f"batch_{i:05d}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in reqs[i:i + CHUNK])
    print(f"{len(reqs)} requests in {work}")


def submit(client, work):
    jobs = []
    for path in sorted(work.glob("batch_*.jsonl")):
        file = client.files.create(file=open(path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file.id, endpoint="/v1/responses", completion_window="24h"
        )
        jobs.append({"file": path.name, "batch_id": batch.id})
    (work / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"submitted {len(jobs)} batches")


def collect(client, work, args):
    key = "caption" if args.stage == "summarize" else "corrected_caption"
    rows, pending = [], 0
    for job in json.loads((work / "jobs.json").read_text()):
        batch = client.batches.retrieve(job["batch_id"])
        if batch.status != "completed":
            pending += 1
            continue
        for line in client.files.content(batch.output_file_id).text.splitlines():
            res = json.loads(line)
            body = res["response"]["body"]
            out = next(c["text"] for o in body["output"] if o["type"] == "message"
                       for c in o["content"] if c["type"] == "output_text")
            rows.append({"filename": res["custom_id"], "caption": json.loads(out)[key]})
    if pending:
        print(f"{pending} batches not completed yet")
    name = "gpt41" if args.stage == "summarize" else "long"
    dst = Path("data/captions") / f"{name}_{args.split}.csv"
    pd.DataFrame(rows).to_csv(dst, index=False)
    print(f"wrote {len(rows)} captions to {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["build", "submit", "collect"])
    p.add_argument("--stage", choices=["correct", "summarize"], required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--data-root")
    p.add_argument("--drafts", help="input captions csv (filename,caption)")
    p.add_argument("--work-dir", default="gpt_batches")
    args = p.parse_args()

    work = Path(args.work_dir) / args.stage / args.split
    if args.action == "build":
        build(args, work)
    elif args.action == "submit":
        submit(OpenAI(), work)
    else:
        collect(OpenAI(), work, args)
