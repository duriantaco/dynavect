import torch

CLIP_MEAN = None
CLIP_STD = None

def clip_preprocess(x, device):
    global CLIP_MEAN, CLIP_STD
    if CLIP_MEAN is None:
        CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
    x = (x + 1) / 2
    return (x - CLIP_MEAN) / CLIP_STD
