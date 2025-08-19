import os, sys

CANDIDATES = [
    os.path.join(os.path.dirname(__file__), 'stylegan2-ada-pytorch'),
    os.path.join(os.path.dirname(__file__), '..', 'stylegan2-ada-pytorch'),
    os.environ.get('STYLEGAN2_ADA_DIR', ''),
]
for p in CANDIDATES:
    if p and os.path.isdir(p):
        p = os.path.abspath(p)
        if p not in sys.path:
            sys.path.insert(0, p)
        break


# stylegan2_repo_path = './stylegan2-ada-pytorch'
# if stylegan2_repo_path not in sys.path:
#     sys.path.append(stylegan2_repo_path)

import clip
import dnnlib, legacy

def load_stylegan2(device):
    print("Loading StyleGAN2-FFHQ model...")
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    with dnnlib.util.open_url(url) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print("StyleGAN2 model loaded.")
    return G

def load_clip(device):
    print("Loading CLIP model...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    print("CLIP model loaded.")
    return model
