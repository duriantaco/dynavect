import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image
import re
import torch
import torch.nn.functional as F
import clip
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.optim import Adam
import lpips
import pandas as pd
from new import DynaVect, EditEvaluator, GlobalDirectionBank

def load_stylegan2(device):
    print("Loading StyleGAN2-FFHQ model...")
    import dnnlib
    import legacy
    
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

class StyleCLIP_Optimizer:

    def __init__(self, G, clip_model, device, optimization_steps=100):
        self.G = G
        self.clip_model = clip_model
        self.device = device
        self.optimization_steps = optimization_steps
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.use_w_plus = 'DynaVect' in globals() and DynaVect(G, clip_model, device).use_w_plus

    def __call__(self, w_latent, edits):
        w_current = w_latent.clone()
        with torch.no_grad():
            img_original = self.G.synthesis(w_latent, noise_mode='const')

        for edit in edits:
            print(f"\nOptimizing for: '{edit['target']}'")
            w_current = self._optimize(w_current, img_original, edit['target'])

        return w_current, None 

    def _optimize(self, w_start, img_original, target_text):
        target_tokens = clip.tokenize([target_text]).to(self.device)
        with torch.no_grad():
            target_features = self.clip_model.encode_text(target_tokens).float()
            target_features /= target_features.norm(dim=-1, keepdim=True)

        w_opt = w_start.clone().detach().requires_grad_(True)
        optimizer = Adam([w_opt], lr=0.01)

        pbar = tqdm(range(self.optimization_steps), desc=f"StyleCLIP Optimizing '{target_text}'")
        for _ in pbar:
            optimizer.zero_grad()
            img_edited = self.G.synthesis(w_opt, noise_mode='const')

            img_resized = F.interpolate(img_edited, size=(224, 224), mode='bilinear', align_corners=False)
            img_features = self.clip_model.encode_image(img_resized)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            clip_loss = 1 - torch.mean(F.cosine_similarity(img_features, target_features))

            identity_loss = self.lpips_loss_fn(img_original, img_edited).mean()

            latent_loss = F.mse_loss(w_start, w_opt)

            total_loss = clip_loss + 0.8 * identity_loss + 0.5 * latent_loss

            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "CLIP": f"{clip_loss.item():.4f}",
                "LPIPS": f"{identity_loss.item():.4f}"
            })

        return w_opt.detach()

class GlobalDirectionOnly:

    def __init__(self, G, clip_model, device):
        self.direction_bank = GlobalDirectionBank(G, clip_model, device)
        self.use_w_plus = self.direction_bank.use_w_plus

    def __call__(self, w_latent, edits):
        w_current = w_latent.clone()
        for edit in edits:
            direction_data = self.direction_bank.get_direction(
                edit['neutral'], edit['target'], mode='sample'
            )
            w_current += direction_data['base'] * edit.get('strength', 1.0)
        return w_current, None

def run_sota_comparison(G, clip_model, device):

    print("\n" + "="*80)
    print("RUNNING COMPARISON")
    print("="*80)

    print("\n Initializing editing methods...")
    dynavect_editor = DynaVect(G, clip_model, device)
    try:
        checkpoint = torch.load("enhanced_modulator_checkpoint.pt", map_location=device)
        dynavect_editor.modulator.load_state_dict(checkpoint['model_state_dict'])
        print("DynaVect (Our Method) loaded from checkpoint.")
    except FileNotFoundError:
        print("Warning: Could not load checkpoint for DynaVect. It will run with random weights.")
        return

    styleclip_optimizer = StyleCLIP_Optimizer(G, clip_model, device)
    print("StyleCLIP (Optimizer) initialized.")
    global_direction_editor = GlobalDirectionOnly(G, clip_model, device)
    print("StyleCLIP (Global Direction) initialized.")

    methods = {
        "DynaVect (Ours)": lambda w, e: dynavect_editor.edit_combined(w, e, preserve_attributes=['identity', 'gender']),
        "StyleCLIP (Optimizer)": styleclip_optimizer,
        "StyleCLIP (Global Dir)": global_direction_editor
    }

    print("\n[2/4] Defining test cases...")
    test_edits = [
        {"neutral": "a face", "target": "a smiling face", "strength": 0.8},
        {"neutral": "a young person", "target": "an old person", "strength": 0.7},
        {"neutral": "a person with dark hair", "target": "a person with blonde hair", "strength": 0.9}
    ]
    num_test_runs = 10
    sota_test_cases = [{'edits': test_edits} for _ in range(num_test_runs)]
    print(f"Prepared {num_test_runs} test runs with the edits: Smile, Age, and Hair Color.")

    print("\n[3/4] Running evaluation loop...")
    evaluator = EditEvaluator(device)
    sota_results = {}

    for method_name, method_fn in methods.items():
        print("\n" + "-"*50)
        print(f"Evaluating: {method_name}")
        print("-"*50)
        results = evaluator.evaluate_method(method_fn, sota_test_cases, G, clip_model)
        sota_results[method_name] = results

    print("\n[4/4] Generating results summary...")
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Method':<25} | {'CLIP Score (↑)':<20} | {'LPIPS (↓)':<20} | {'L2 Distance (↓)':<20}")
    print("-"*85)

    report_data = []
    for method_name, results in sota_results.items():
        clip_score = f"{results['clip_score']['mean']:.4f} ± {results['clip_score']['std']:.4f}"
        lpips_score = f"{results['lpips']['mean']:.4f} ± {results['lpips']['std']:.4f}"
        l2_score = f"{results['l2_distance']['mean']:.4f} ± {results['l2_distance']['std']:.4f}"
        print(f"{method_name:<25} | {clip_score:<20} | {lpips_score:<20} | {l2_score:<20}")

        report_data.append({
            "Method": method_name,
            "CLIP Score Mean": results['clip_score']['mean'],
            "CLIP Score Std": results['clip_score']['std'],
            "LPIPS Mean": results['lpips']['mean'],
            "LPIPS Std": results['lpips']['std'],
            "L2 Distance Mean": results['l2_distance']['mean'],
            "L2 Distance Std": results['l2_distance']['std'],
        })

    results_df = pd.DataFrame(report_data)
    os.makedirs("sota_results", exist_ok=True)
    results_path = "sota_results/sota_comparison_metrics.csv"
    results_df.to_csv(results_path, index=False)
    print("-"*85)
    print(f"\nDetailed results table saved to: {results_path}")
    print("="*80)


def slugify(text):
    """Helper function to create a clean filename from a method name."""
    text = text.lower()
    return re.sub(r'[\s\(\)]+', '_', text).strip('_')

def generate_comparison_images(G, clip_model, device):
    print("\n" + "="*80)
    print("GENERATING VISUAL COMPARISON IMAGES")
    print("="*80)

    seeds = [42, 123, 888, 1024, 2023]
    output_dir = "visual_comparison"
    os.makedirs(output_dir, exist_ok=True)

    edits = [
        {"neutral": "a face", "target": "a smiling face", "strength": 0.8},
        {"neutral": "a young person", "target": "an old person", "strength": 0.7},
        {"neutral": "a person with dark hair", "target": "a person with blonde hair", "strength": 0.9}
    ]
    edit_name = "smile_age_hair"
    print(f"Generating images for {len(seeds)} seeds with the edit: '{edit_name}'")

    print("\nInitializing editing methods...")
    dynavect_editor = DynaVect(G, clip_model, device)
    try:
        checkpoint = torch.load("enhanced_modulator_checkpoint.pt", map_location=device)
        dynavect_editor.modulator.load_state_dict(checkpoint['model_state_dict'])
        print("DynaVect (Ours) loaded from checkpoint.")
    except FileNotFoundError:
        print("Could not load checkpoint for DynaVect.")
        return

    methods = {
        "DynaVect (Ours)": lambda w, e: dynavect_editor.edit_combined(w, e, preserve_attributes=['identity', 'gender']),
        "StyleCLIP (Optimizer)": StyleCLIP_Optimizer(G, clip_model, device),
        "StyleCLIP (Global Dir)": GlobalDirectionOnly(G, clip_model, device)
    }

    for seed in tqdm(seeds, desc="Processing Seeds"):
        torch.manual_seed(seed)

        z = torch.randn(1, G.z_dim).to(device)
        w_source = G.mapping(z, None, truncation_psi=0.7)
        if len(w_source.shape) == 2:
            w_source = w_source.unsqueeze(1).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            original_img = G.synthesis(w_source, noise_mode='const')

        save_image((original_img + 1) / 2, os.path.join(output_dir, f"seed_{seed}_original.png"))

        for method_name, method_fn in methods.items():
            print(f"\nApplying method '{method_name}' for seed {seed}...")
            w_edited, _ = method_fn(w_source.clone(), edits)

            with torch.no_grad():
                edited_img = G.synthesis(w_edited, noise_mode='const')

            clean_name = slugify(method_name)
            save_path = os.path.join(output_dir, f"seed_{seed}_edit_{edit_name}_{clean_name}.png")
            save_image((edited_img + 1) / 2, save_path)
            print(f" Saved image to {save_path}")

    print("\n" + "="*80)
    print(f"All images generated successfully in the '{output_dir}' folder!")
    print("="*80)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    G = load_stylegan2(DEVICE)
    clip_model = load_clip(DEVICE)

    generate_comparison_images(G, clip_model, DEVICE)