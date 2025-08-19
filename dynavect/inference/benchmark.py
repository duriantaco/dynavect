import os, json, torch, clip, lpips, pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from loaders import load_stylegan2
from dynavect import DynaVect
from direction_bank import GlobalDirectionBank
from evaluation.evaluator import EditEvaluator
from clip_utils import clip_preprocess

class StyleCLIP_Optimizer:
    def __init__(self, G, clip_model, device, steps=100):
        self.G, self.clip_model, self.device, self.steps = G, clip_model, device, steps
        self.lpips = lpips.LPIPS(net='alex').to(device)
        self.use_w_plus = True

    def __call__(self, w_latent, edits):
        with torch.no_grad():
            img_ref = self.G.synthesis(w_latent, noise_mode='const')
        target_text = ", ".join([e['target'] for e in edits])
        w_opt = self._optimize(w_latent, img_ref, target_text)
        return w_opt, None

    def _optimize(self, w_start, img_ref, target_text):
        toks = clip.tokenize([target_text]).to(self.device)
        with torch.no_grad():
            tfeat = self.clip_model.encode_text(toks).float()
            tfeat = F.normalize(tfeat, dim=-1)
        w = w_start.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=1e-2)
        
        for _ in range(self.steps):
            opt.zero_grad()
            img = self.G.synthesis(w, noise_mode='const')
            img224 = F.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
            img224 = clip_preprocess(img224, self.device)

            imfeat = self.clip_model.encode_image(img224).float()
            imfeat = F.normalize(imfeat, dim=-1)

            clip_loss = 1 - (imfeat * tfeat).sum(dim=-1)
            clip_loss = clip_loss.mean()

            id_loss = self.lpips(img_ref, img).mean()
            lat_loss = F.mse_loss(w_start, w)
            (clip_loss + 0.8*id_loss + 0.5*lat_loss).backward()
            opt.step()
        return w.detach()

class GlobalDirectionOnly:
    def __init__(self, G, clip_model, device, use_w_plus=True):
        self.bank = GlobalDirectionBank(G, clip_model, device, use_w_plus)
        self.use_w_plus = use_w_plus

    def __call__(self, w_latent, edits):
        w = w_latent.clone()
        for e in edits:
            d = self.bank.get_direction(e['neutral'], e['target'], mode='sample')['base']
            w = w + d * e.get('strength', 1.0)
        return w, None

def run_sota_comparison(G, clip_model, device):
    dynavect = DynaVect(G, clip_model, device)
    ckpt_path = "enhanced_modulator_checkpoint_v1.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        dynavect.modulator.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {ckpt_path}")
    else:
        raise FileNotFoundError(f"Missing {ckpt_path}")

    def meth_dynavect(w, e):
        return dynavect.edit_combined(
            w, 
            e, 
            preserve_attributes=['identity']
        )
    
    meth_opt = StyleCLIP_Optimizer(G, clip_model, device, steps=100)
    meth_gdir = GlobalDirectionOnly(G, clip_model, device, use_w_plus=True)

    evaluator = EditEvaluator(device, prefer_pack="antelopev2")

    seeds = [42,95,101,683,909,1138,1492,1776,2001,2077]
    cases = [
    {"name":"Smile", "edits":[{"neutral":"a face","target":"a smiling face","strength":0.9}]},
    {"name":"Age+", "edits":[{"neutral":"a face","target":"an old person","strength":0.8}]},
    {"name":"Blonde", "edits":[{"neutral":"a face","target":"a person with blonde hair","strength":1.1}]},

    {"name":"Combo-NonOcc","edits":[
        {"neutral":"a face","target":"a smiling face","strength":0.8},
        {"neutral":"a face","target":"an old person","strength":0.7},
        {"neutral":"a face","target":"a person with blonde hair","strength":1.0},
    ]},

    {"name":"Combo-Occ","edits":[
        {"neutral":"a face","target":"a smiling face","strength":0.8},
        {"neutral":"a face","target":"an old person","strength":0.7},
        {"neutral":"a face","target":"a person wearing sunglasses","strength":1.2},
    ]},
]

    methods = {
        "DynaVect": meth_dynavect,
        "StyleCLIP-O": meth_opt,
        "StyleCLIP-G": meth_gdir,
    }

    per_run = []
    for seed in tqdm(seeds, desc="Seeds"):
        torch.manual_seed(seed)
        z = torch.randn(1, G.z_dim, device=device)
        w0 = G.mapping(z, None, truncation_psi=0.7)
        if w0.dim()==2: 
            w0 = w0.unsqueeze(1).repeat(1, G.num_ws, 1)
        with torch.no_grad():
            img0 = G.synthesis(w0, noise_mode='const')

        for case in cases:
            edits = case['edits']
            targets = [e['target'] for e in edits]
            target_text = ", ".join(targets)

            neutral_txt = 'a face'
            for mname, mfn in methods.items():
                w1,_ = mfn(w0.clone(), case["edits"])
                with torch.no_grad():
                    img1 = G.synthesis(w1, noise_mode='const')
                m = evaluator.evaluate_edit(img0, img1, target_text, clip_model, neutral_text=neutral_txt)
                m.update({"method": mname, "case": case["name"], "seed": seed})
                per_run.append(m)

    df = pd.DataFrame(per_run)
    g = df.groupby(["method","case"])
    agg = g.agg({
        "clip_dir":["mean","std"],
        "identity":["mean","std"],
        "lpips":["mean","std"],
        "l2_distance":["mean","std"],
        "identity_detected":["mean"]
    }).reset_index()
    agg.columns = ["method","case",
                   "ΔCLIP_mean","ΔCLIP_std",
                   "ID_mean","ID_std",
                   "LPIPS_mean","LPIPS_std",
                   "L2_mean","L2_std",
                   "ArcFace_rate"]

    os.makedirs("sota_results", exist_ok=True)
    agg.to_csv("sota_results/benchmark_summary.csv", index=False)
    with open("sota_results/benchmark_runs.json","w") as f:
        json.dump(per_run, f, indent=2)

    print("\n BENCHMARK (mean ± std) ")
    for (_, _), row in agg.groupby(["method","case"]):
        r = row.iloc[0]
        print(f"{r.method:12s} | {r.case:7s} | ΔCLIP {r.ΔCLIP_mean:.4f}±{r.ΔCLIP_std:.4f} | "
              f"ID {r.ID_mean:.3f}±{r.ID_std:.3f} | LPIPS {r.LPIPS_mean:.3f}±{r.LPIPS_std:.3f} | "
              f"L2 {r.L2_mean:.4f}±{r.L2_std:.4f} | ArcFace% {100*r.ArcFace_rate:.1f}")

def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    G = load_stylegan2(DEVICE)
    clip_model = load_clip(DEVICE)
    run_sota_comparison(G, clip_model, DEVICE)
