"""Prototypical network (Section 4.3, Figure 1).

The image encoder of an OpenCLIP model is trained episodically: prototypes are
the mean support embedding of each class, and queries are scored by the
negative squared Euclidean distance to every prototype.
"""

import open_clip
import torch
import torch.nn as nn

ENCODERS = {
    # name: (open_clip architecture, pretrained tag, input size)
    "bioclip": ("hf-hub:imageomics/bioclip", None, 224),
    "siglip-512": ("ViT-B-16-SigLIP-512", "webli", 512),
}


class ProtoNet(nn.Module):
    def __init__(self, encoder, pretrained=True):
        super().__init__()
        arch, tag, _ = ENCODERS[encoder]
        clip = open_clip.create_model(arch, pretrained=tag if pretrained else None)
        self.encoder = clip.visual

    def encode(self, images):
        return self.encoder(images)

    def forward(self, support, query):
        """support: (K, S, C, H, W), query: (Q, C, H, W) -> logits (Q, K)."""
        k, s = support.shape[:2]
        with torch.autocast(device_type=support.device.type):
            protos = self.encoder(support.flatten(0, 1)).view(k, s, -1).mean(dim=1)
            q = self.encoder(query)
            return -torch.cdist(q.float(), protos.float()).pow(2)
