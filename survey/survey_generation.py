import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import os
import sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import argparse

stylegan2_repo_path = './stylegan2-ada-pytorch'
if stylegan2_repo_path not in sys.path:
    sys.path.append(stylegan2_repo_path)
import lpips

def load_stylegan2(device):
    print("Loading StyleGAN2-FFHQ model...")
    import dnnlib, legacy
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    with dnnlib.util.open_url(url) as f: G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print("StyleGAN2 model loaded.")
    return G

def load_clip(device):
    print("Loading CLIP model...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    print("CLIP model loaded.")
    return model

def tensor_to_pil(tensor):
    tensor = (tensor.clone().detach().squeeze(0).permute(1, 2, 0) + 1.) / 2.
    tensor = (tensor * 255).clip(0, 255).to(torch.uint8)
    return Image.fromarray(tensor.cpu().numpy(), 'RGB')

def get_font(size, bold=False):
    font_paths = ["C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for font_path in font_paths:
        try:
            if bold and "arialbd.ttf" in font_path:
                return ImageFont.truetype(font_path, size)
            elif not bold and "arial.ttf" in font_path:
                 return ImageFont.truetype(font_path, size)
            return ImageFont.truetype(font_path, size)
        except IOError: continue
    return ImageFont.load_default()

class GlobalDirectionBank:
    def __init__(self, G, clip_model, device, use_w_plus=True):
        self.G = G
        self.clip_model = clip_model
        self.device = device
        self.use_w_plus = use_w_plus
        self.w_dim = G.w_dim
        self.num_layers = G.num_ws
        self.directions_cache = {}

    def get_direction(self, neutral_text, target_text, mode='sample'):
        if mode == 'sample':
            cache_key = (neutral_text, target_text, 'sample')
            if cache_key in self.directions_cache:
                return self.directions_cache[cache_key]
            direction = self._find_direction_sampling(neutral_text, target_text)
            self.directions_cache[cache_key] = {'base': direction}
            return self.directions_cache[cache_key]
        else:
            raise ValueError("Mode must be 'sample'")

    def _find_direction_sampling(self, neutral_text, target_text, num_samples=2000):
        with torch.no_grad():
            neutral_tokens = clip.tokenize([neutral_text]).to(self.device)
            target_tokens = clip.tokenize([target_text]).to(self.device)
            neutral_features = self.clip_model.encode_text(neutral_tokens).float()
            target_features = self.clip_model.encode_text(target_tokens).float()
            z = torch.randn(num_samples, self.G.z_dim).to(self.device)
            w = self.G.mapping(z, None, truncation_psi=0.7)
            if self.use_w_plus and len(w.shape) == 2:
                w = w.unsqueeze(1).repeat(1, self.num_layers, 1)
            
            batch_size = 32
            neutral_scores, target_scores = [], []
            for i in range(0, num_samples, batch_size):
                batch_w = w[i:i + batch_size]
                imgs_resized = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
                img_features = self.clip_model.encode_image(imgs_resized).float()
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                neutral_scores.append(F.cosine_similarity(img_features, neutral_features))
                target_scores.append(F.cosine_similarity(img_features, target_features))
            
            neutral_scores = torch.cat(neutral_scores)
            target_scores = torch.cat(target_scores)
            top_k = 20
            neutral_indices = torch.topk(neutral_scores, top_k).indices
            target_indices = torch.topk(target_scores, top_k).indices
            direction = w[target_indices].mean(dim=0) - w[neutral_indices].mean(dim=0)
            return direction

class DynamicContextualModulator(nn.Module):
    def __init__(self, w_dim=512, clip_dim=512, num_layers=18):
        super().__init__()
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(clip_dim * 2, 2048), nn.LayerNorm(2048), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(2048, 2048), nn.LayerNorm(2048), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.2),
        )
        self.layer_heads = nn.ModuleList([nn.Linear(1024, w_dim) for _ in range(num_layers)])
        self.layer_attention = nn.Sequential(nn.Linear(1024, num_layers), nn.Softmax(dim=-1))

    def forward(self, source_img_features, target_text_features):
        context = torch.cat([source_img_features, target_text_features], dim=-1)
        features = self.net(context)
        layer_deltas = torch.stack([head(features) for head in self.layer_heads], dim=1)
        attention_weights = self.layer_attention(features).unsqueeze(-1)
        return layer_deltas * attention_weights

class DynaVect:
    def __init__(self, G, clip_model, device):
        self.G = G
        self.clip_model = clip_model
        self.device = device
        self.use_w_plus = True
        self.direction_bank = GlobalDirectionBank(G, clip_model, device, self.use_w_plus)
        self.modulator = DynamicContextualModulator(w_dim=G.w_dim, num_layers=G.num_ws).to(device)
        self.disentanglement_bases = {}
        self._initialize_disentanglement_bases()

    def _initialize_disentanglement_bases(self):
        base_attributes = [
            ("identity", [("a face", "another face")]),
            ("gender", [("a woman's face", "a man's face")]),
            ("age", [("a young person", "an old person")]),
            ("expression", [("a neutral face", "a smiling face")]),
            ("race", [("an asian face", "a caucasian face"), ("a black face", "a caucasian face")])
        ]
        for attr_name, prompts in base_attributes:
            self.disentanglement_bases[attr_name] = [self.direction_bank.get_direction(n, t, mode='sample')['base'] for n, t in prompts]

    def _get_image_features(self, w_latent):
        with torch.no_grad():
            img = self.G.synthesis(w_latent, noise_mode='const')
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img_features = self.clip_model.encode_image(img_resized).float()
            return img_features / img_features.norm(dim=-1, keepdim=True)

    def _get_text_features(self, text):
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            text_features = self.clip_model.encode_text(tokens).float()
            return text_features / text_features.norm(dim=-1, keepdim=True)

    def _orthogonalize(self, direction, preserve_attributes):
        if not preserve_attributes: return direction
        direction_flat = direction.view(1, -1)
        for attr_name in preserve_attributes:
            if attr_name in self.disentanglement_bases:
                for basis in self.disentanglement_bases[attr_name]:
                    basis_flat = basis.view(1, -1)
                    proj = torch.sum(direction_flat * basis_flat) / torch.sum(basis_flat * basis_flat) * basis_flat
                    direction_flat -= proj
        return direction_flat.view(direction.shape)

    def edit_combined(self, w_latent, edits, preserve_attributes=['identity', 'gender', 'race']):
        self.modulator.eval()
        w_current = w_latent.clone()
        for edit_params in edits:
            neutral_text, target_text, strength, mod_strength = edit_params["neutral"], edit_params["target"], edit_params.get("strength", 1.0), edit_params.get("modulation_strength", 0.7)
            
            baseline_direction = self.direction_bank.get_direction(neutral_text, target_text, mode='sample')['base']
            
            with torch.no_grad():
                source_img_features = self._get_image_features(w_current)
                target_text_features = self._get_text_features(target_text)
                predicted_delta = self.modulator(source_img_features, target_text_features)

            final_direction = baseline_direction + predicted_delta * mod_strength
            final_direction = self._orthogonalize(final_direction, preserve_attributes)
            w_current += final_direction * strength
        return w_current, None

class StyleCLIP_Optimizer:
    def __init__(self, G, clip_model, device, optimization_steps=100):
        self.G = G; self.clip_model = clip_model; self.device = device; self.optimization_steps = optimization_steps
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
    
    def __call__(self, w_latent, edits):
        w_current = w_latent.clone()
        with torch.no_grad(): img_original = self.G.synthesis(w_latent, noise_mode='const')
        combined_target = ", ".join([edit['target'] for edit in edits])
        w_current = self._optimize(w_current, img_original, combined_target)
        return w_current, None

    def _optimize(self, w_start, img_original, target_text):
        target_tokens = clip.tokenize([target_text]).to(self.device)
        with torch.no_grad():
            target_features = self.clip_model.encode_text(target_tokens).float()
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)
        w_opt = w_start.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w_opt], lr=0.01)
        identity_loss_weight = 0.8
        if "sunglasses" in target_text or "glasses" in target_text: identity_loss_weight = 0.2 
        for _ in range(self.optimization_steps):
            optimizer.zero_grad()
            img_edited = self.G.synthesis(w_opt, noise_mode='const')
            img_resized = F.interpolate(img_edited, size=(224, 224), mode='bilinear', align_corners=False)
            img_features = self.clip_model.encode_image(img_resized)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            clip_loss = 1 - torch.mean(F.cosine_similarity(img_features, target_features))
            identity_loss = self.lpips_loss_fn(img_original, img_edited).mean()
            latent_loss = F.mse_loss(w_start, w_opt)
            total_loss = clip_loss + identity_loss_weight * identity_loss + 0.5 * latent_loss
            total_loss.backward(); optimizer.step()
        return w_opt.detach()

class GlobalDirectionOnly:
    def __init__(self, G, clip_model, device):
        self.direction_bank = GlobalDirectionBank(G, clip_model, device)
    
    def __call__(self, w_latent, edits):
        w_current = w_latent.clone()
        for edit in edits:
            direction_data = self.direction_bank.get_direction(edit['neutral'], edit['target'], mode='sample')
            w_current += direction_data['base'] * edit.get('strength', 1.0)
        return w_current, None

def generate_survey_images(G, clip_model, device, checkpoint_path, seeds, edits):
    print("\n" + "="*80 + "\n GENERATING UNLABELED IMAGES FOR USER SURVEY\n" + "="*80)
    output_dir = "survey_materials"
    os.makedirs(output_dir, exist_ok=True)

    dynavect_editor = DynaVect(G, clip_model, device)
    try:
        dynavect_editor.modulator.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"FATAL ERROR loading checkpoint: {e}"); return

    methods = {
        "ours": dynavect_editor.edit_combined,
        "styleclip_o": StyleCLIP_Optimizer(G, clip_model, device),
        "styleclip_g": GlobalDirectionOnly(G, clip_model, device)
    }

    for seed in tqdm(seeds, desc="Processing Seeds for Survey"):
        torch.manual_seed(seed)
        z = torch.randn(1, G.z_dim).to(device)
        w_source = G.mapping(z, None, truncation_psi=0.7)
        if len(w_source.shape) == 2:
            w_source = w_source.unsqueeze(1).repeat(1, G.num_ws, 1)
        
        with torch.no_grad():
            original_img_tensor = G.synthesis(w_source, noise_mode='const')
        original_pil = tensor_to_pil(original_img_tensor)
        original_pil.save(os.path.join(output_dir, f"seed_{seed}_original.png"))

        for edit_name, edit_config in edits.items():
            for method_name, method_fn in methods.items():
                try:
                    w_edited, _ = method_fn(w_source.clone(), edit_config["edits"])
                    with torch.no_grad():
                        edited_img_tensor = G.synthesis(w_edited, noise_mode='const')
                    edited_pil = tensor_to_pil(edited_img_tensor)
                    filename = f"seed_{seed}_edit_{edit_name}_method_{method_name}.png"
                    edited_pil.save(os.path.join(output_dir, filename))
                except Exception as e:
                    print(f"Failed: seed {seed}, edit {edit_name}, method {method_name}. Error: {e}")
    
    print(f"\nSuccess! All survey images saved in '{output_dir}'.")

def generate_reference_grids(G, clip_model, device, checkpoint_path, seeds, edits):
    print("\n" + "="*80 + "\n GENERATING LABELED GRIDS FOR YOUR REFERENCE\n" + "="*80)
    output_dir = "reference_grids"
    os.makedirs(output_dir, exist_ok=True)

    dynavect_editor = DynaVect(G, clip_model, device)
    try:
        dynavect_editor.modulator.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    except Exception as e:
        print(f"FATAL ERROR loading checkpoint: {e}"); return
    
    methods = {
        "DynaVect (Ours)": dynavect_editor.edit_combined,
        "StyleCLIP-O": StyleCLIP_Optimizer(G, clip_model, device),
        "StyleCLIP-G": GlobalDirectionOnly(G, clip_model, device)
    }

    for seed in tqdm(seeds, desc="Processing Seeds for Grids"):
        torch.manual_seed(seed)
        z = torch.randn(1, G.z_dim).to(device)
        w_source = G.mapping(z, None, truncation_psi=0.7)
        if len(w_source.shape) == 2:
            w_source = w_source.unsqueeze(1).repeat(1, G.num_ws, 1)

        img_s, pad, header_h, label_w = 256, 20, 80, 200
        num_methods = len(methods) + 1
        num_edits = len(edits)
        
        grid_w = label_w + num_methods * (img_s + pad) + pad
        grid_h = header_h + num_edits * (img_s + pad) + pad
        fig = Image.new('RGB', (grid_w, grid_h), 'white')
        draw = ImageDraw.Draw(fig)
        header_font = get_font(20, bold=True)
        label_font = get_font(18)

        all_headers = ["Original"] + list(methods.keys())
        for i, name in enumerate(all_headers):
            bbox = draw.textbbox((0,0), name, font=header_font)
            text_w = bbox[2] - bbox[0]
            draw.text((label_w + i * (img_s + pad) + (img_s - text_w) / 2, (header_h - (bbox[3] - bbox[1])) / 2), name, fill="black", font=header_font)

        for edit_idx, (edit_name, edit_config) in enumerate(edits.items()):
            y = header_h + edit_idx * (img_s + pad)
            
            bbox_label = draw.textbbox((0,0), edit_name, font=label_font)
            draw.text((pad, y + (img_s - (bbox_label[3] - bbox_label[1])) / 2), edit_name, fill="black", font=label_font)
            
            with torch.no_grad():
                original_img_tensor = G.synthesis(w_source, noise_mode='const')
            original_pil = tensor_to_pil(original_img_tensor).resize((img_s, img_s), Image.Resampling.LANCZOS)
            fig.paste(original_pil, (label_w + pad, y))

            for method_idx, (method_name, method_fn) in enumerate(methods.items()):
                x = label_w + (method_idx + 1) * (img_s + pad) + pad
                try:
                    w_edited, _ = method_fn(w_source.clone(), edit_config["edits"])
                    with torch.no_grad():
                        edited_img_tensor = G.synthesis(w_edited, noise_mode='const')
                    edited_pil = tensor_to_pil(edited_img_tensor).resize((img_s, img_s), Image.Resampling.LANCZOS)
                    fig.paste(edited_pil, (x, y))
                except Exception as e:
                     fig.paste(Image.new('RGB', (img_s, img_s), 'black'), (x, y))

        fig_path = os.path.join(output_dir, f"reference_grid_seed_{seed}.png")
        fig.save(fig_path)

    print(f"\nSuccess! All reference grids saved in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DynaVect Survey & Reference Generation")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'survey', 'reference'], help="Which task to run.")
    parser.add_argument('--checkpoint_path', type=str, default='enhanced_modulator_checkpoint_v2.pt', help='Path to your trained model checkpoint')
    args = parser.parse_args()
    
    seeds_for_survey = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                        180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                        520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538,
                        121, 671,  993, 1121, 158, 1721, 2016, 2099, 3212, 4127, 645, 313, 1562, 1567,
                        2526, 7272, 15121, 494, 202, 888, 4020, 27, 95, 101, 683,  909,  1138, 1492, 1776, 
                        2001, 2077, 3000, 4126, 4540, 8912, 9999, 1892, 1131, 4359, 12345, 11378, 15678,
                        123455, 11121, 22222, 11414, 15151, 1123, 5516, 6162, 8284, 2724, 7245, 7238, 
                        9678, 1783, 2847, 8345, 8222, 945, 2845, 5161, 734, 282, 3427, 2828, 8932, 349]
    
    edits_for_survey = {
        "age_up": {"edits": [{"neutral": "a young person", "target": "an old person", "strength": 1.4}]},
        "age_down": {"edits": [{"neutral": "an old person", "target": "a young person", "strength": 1.4}]},
        "sad_face": {"edits": [{"neutral": "a face", "target": "a sad face", "strength": 1.5}]},
        "angry_face": {"edits": [{"neutral": "a face", "target": "an angry face", "strength": 1.5}]},
        "combined": { "edits": [
            {"neutral": "a face", "target": "a person wearing sunglasses", "strength": 1.4},
            {"neutral": "a young person", "target": "an old person", "strength": 1.4},
            {"neutral": "a face", "target": "a smiling face", "strength": 1.4},
        ]}
    }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    G = load_stylegan2(DEVICE)
    clip_model = load_clip(DEVICE)
    
    if args.mode == 'all' or args.mode == 'survey':
        generate_survey_images(G, clip_model, DEVICE, args.checkpoint_path, seeds_for_survey, edits_for_survey)

    if args.mode == 'all' or args.mode == 'reference':
        generate_reference_grids(G, clip_model, DEVICE, args.checkpoint_path, seeds_for_survey, edits_for_survey)
    
    print("\nPipeline finished.")