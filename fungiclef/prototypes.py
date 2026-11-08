"""Prototype construction and top-5 retrieval shared by every method.

Prototype of a class = mean of the L2-normalised embeddings of its support
images. A test observation (one or more images) is embedded the same way and
ranked against all prototypes by cosine similarity (Sections 4.1 and 4.3).
"""

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def mean_embedding(encode, transform, paths, device):
    batch = torch.stack([transform(p) for p in paths]).to(device)
    emb = F.normalize(encode(batch).float(), dim=1)
    return emb.mean(dim=0)


def build_prototypes(encode, transform, support_df, device):
    ids, protos = [], []
    for cid, group in tqdm(support_df.groupby("category_id"), desc="prototypes"):
        ids.append(int(cid))
        protos.append(mean_embedding(encode, transform, group["path"].tolist(), device))
    return ids, torch.stack(protos)


def predict_top5(encode, transform, test_df, ids, protos, device):
    ids = torch.tensor(ids)
    rows = []
    for obs_id, group in tqdm(test_df.groupby("observationID", sort=False), desc="test"):
        q = mean_embedding(encode, transform, group["path"].tolist(), device)
        sim = F.cosine_similarity(q.unsqueeze(0), protos, dim=1)
        top5 = ids[sim.topk(5).indices.cpu()].tolist()
        rows.append({"observationId": obs_id, "predictions": " ".join(map(str, top5))})
    return pd.DataFrame(rows)
