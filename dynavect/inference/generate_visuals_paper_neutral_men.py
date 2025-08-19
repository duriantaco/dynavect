import torch
import torch.nn.functional as F
import clip
import numpy as np
import os
import sys
from tqdm import tqdm
from torchvision.utils import save_image
import re
import lpips
from PIL import Image, ImageDraw, ImageFont
from new import DynaVect, GlobalDirectionBank

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
        self.use_w_plus = True

    def __call__(self, w_latent, edits):
        w_current = w_latent.clone()
        with torch.no_grad():
            img_original = self.G.synthesis(w_latent, noise_mode='const')

        combined_target = ", ".join([edit['target'] for edit in edits])
        
        w_current = self._optimize(w_current, img_original, combined_target)
        return w_current, None


    def _optimize(self, w_start, img_original, target_text):
        from torch.optim import Adam
        target_tokens = clip.tokenize([target_text]).to(self.device)
        with torch.no_grad():
            target_features = self.clip_model.encode_text(target_tokens).float()
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)

        w_opt = w_start.clone().detach().requires_grad_(True)
        optimizer = Adam([w_opt], lr=0.01)
        pbar = tqdm(range(self.optimization_steps), desc=f"Optimizing '{target_text}'")
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

def tensor_to_pil(tensor):
    tensor = (tensor.clone().detach().squeeze(0).permute(1, 2, 0) + 1.) / 2.
    tensor = (tensor * 255).clip(0, 255).to(torch.uint8)
    return Image.fromarray(tensor.cpu().numpy(), 'RGB')

def get_font(size, bold=False):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]
    
    for font_path in font_paths:
        try:
            if bold and "arial" in font_path.lower():
                bold_path = font_path.replace("arial.ttf", "arialbd.ttf")
                if os.path.exists(bold_path):
                    return ImageFont.truetype(bold_path, size)
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    try:
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()

def add_figure_labels(draw, grid_width, y_positions, font_size=16):
    label_font = get_font(font_size, bold=False) 
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for i, (label, y_pos) in enumerate(zip(labels, y_positions)):
        if i < len(y_positions):
            draw.text((10, y_pos - 30), label, fill="black", font=label_font)

def generate_single_seed_figure(G, clip_model, device, seed=15678):
    print(f"\n GENERATING SINGLE SEED FIGURE FOR SEED {seed}")
    output_dir = "research_paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    edit_config = {
        "edits": [
            {"neutral": "a face", "target": "a smiling face", "strength": 0.7, "modulation_strength": 1.0},
            {"neutral": "a young person", "target": "an old person", "strength": 0.8, "modulation_strength": 1.0},
            {"neutral": "a person with dark hair", "target": "a person with blonde hair", "strength": 0.9, "modulation_strength": 1.0}
        ],
        "prompt": "An old, smiling person with blonde hair"
    }

    print("Initializing editing methods...")
    dynavect_editor = DynaVect(G, clip_model, device)
    try:
        checkpoint = torch.load("enhanced_modulator_checkpoint_v1.pt", map_location=device)
        dynavect_editor.modulator.load_state_dict(checkpoint['model_state_dict'])
        print("DynaVect (Ours) loaded.")
    except FileNotFoundError:
        print("enhanced_modulator_checkpoint_v1.pt not found")
        return

    methods = {
        "Original": None,
        "DynaVect (Ours)": lambda w, e: dynavect_editor.edit_combined(w, e, preserve_attributes=['identity']),
        "StyleCLIP-O": StyleCLIP_Optimizer(G, clip_model, device),
        "StyleCLIP-G": GlobalDirectionOnly(G, clip_model, device)
    }
    
    image_size = 300
    prompt_header_height = 80
    method_label_height = 50
    padding = 25
    
    prompt_font = get_font(24, bold=True)
    method_font = get_font(18, bold=False)

    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim).to(device)
    w_source = G.mapping(z, None, truncation_psi=0.7)
    if len(w_source.shape) == 2:
        w_source = w_source.unsqueeze(1).repeat(1, G.num_ws, 1)

    with torch.no_grad():
        original_img = G.synthesis(w_source, noise_mode='const')
    original_pil = tensor_to_pil(original_img).resize((image_size, image_size), Image.Resampling.LANCZOS)

    method_images = [original_pil]
    method_names = list(methods.keys())
    
    for method_name, method_fn in list(methods.items())[1:]:
        try:
            w_edited, _ = method_fn(w_source.clone(), edit_config["edits"])
            with torch.no_grad():
                edited_img = G.synthesis(w_edited, noise_mode='const')
            img_pil = tensor_to_pil(edited_img).resize((image_size, image_size), Image.Resampling.LANCZOS)
            method_images.append(img_pil)
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            method_images.append(original_pil)

    num_methods = len(methods)
    fig_width = num_methods * image_size + (num_methods + 1) * padding
    fig_height = prompt_header_height + image_size + method_label_height + 2 * padding
    
    fig = Image.new('RGB', (fig_width, fig_height), 'white')
    draw = ImageDraw.Draw(fig)
    
    prompt_text = edit_config["prompt"]
    bbox = draw.textbbox((0, 0), prompt_text, font=prompt_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    prompt_x = (fig_width - text_width) // 2
    prompt_y = (prompt_header_height - text_height) // 2
    draw.text((prompt_x, prompt_y), prompt_text, fill="black", font=prompt_font)

    for i, (method_name, img_pil) in enumerate(zip(method_names, method_images)):
        x_pos = padding + i * (image_size + padding)
        y_pos = prompt_header_height + padding
        
        fig.paste(img_pil, (x_pos, y_pos))
        
        bbox = draw.textbbox((0, 0), method_name, font=method_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        label_x = x_pos + (image_size - text_width) // 2
        label_y = y_pos + image_size + 10
        draw.text((label_x, label_y), method_name, fill="black", font=method_font)

    figure_path = os.path.join(output_dir, f"single_seed_{seed}_comparison.png")
    fig.save(figure_path, dpi=(300, 300))
    print(f" Saved single seed figure: {figure_path}")
    
def generate_research_paper_figures(G, clip_model, device):
 
    print("\n" + "="*80)
    print(" GENERATING RESEARCH PAPER READY FIGURES")
    print("="*80)

    seeds = [494, 202, 888, 4020, 27]

    output_dir = "research_paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    edit_configs = [
        {"name": "Smile", "description": "Smile", "edits": [{"neutral": "a face", "target": "a smiling face", "strength": 0.9}]},
        {"name": "Sad_Face", "description": "Sad Face", "edits": [{"neutral": "a face", "target": "a sad face", "strength": 1.5}]},
        {"name": "Angry_Face", "description": "Angry Face", "edits": [{"neutral": "a face", "target": "an angry old face", "strength": 1.5}]},
        {"name": "Age", "description": "Young -> Old", "edits": [{"neutral": "a young person", "target": "an old person", "strength": 0.9}]},
        {"name": "Age", "description": "Old -> Young", "edits": [{"neutral": "an old person", "target": "a young person", "strength": 0.9}]},
    ]
    print("\nInitializing editing methods...")
    dynavect_editor = DynaVect(G, clip_model, device)
    try:
        checkpoint = torch.load("enhanced_modulator_checkpoint_v1.pt", map_location=device)
        dynavect_editor.modulator.load_state_dict(checkpoint['model_state_dict'])
        print(" DynaVect loaded.")
    except FileNotFoundError:
        print(" enhanced_modulator_checkpoint_v1.pt not found")
        return

    methods = {
        "Original": None,
        "DynaVect (Ours)": lambda w, e: dynavect_editor.edit_combined(w, e, preserve_attributes=['identity']),
        "StyleCLIP-O": StyleCLIP_Optimizer(G, clip_model, device),
        "StyleCLIP-G": GlobalDirectionOnly(G, clip_model, device)
    }
    
    image_size = 256
    method_header_height = 80 
    edit_label_width = 200
    padding = 20
    figure_label_margin = 50

    title_font = get_font(20, bold=True)
    header_font = get_font(16, bold=False)
    label_font = get_font(14)
    caption_font = get_font(12)

    num_methods = len(methods)
    num_edits = len(edit_configs)
    
    grid_width = figure_label_margin + edit_label_width + num_methods * image_size + (num_methods + 1) * padding
    grid_height = method_header_height + num_edits * (image_size + padding) + padding
    
    main_fig = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(main_fig)
    
    method_names = list(methods.keys())
    for i, method_name in enumerate(method_names):
        x_pos = figure_label_margin + edit_label_width + padding + i * (image_size + padding)
        y_pos = 10
        
        bbox = draw.textbbox((0, 0), method_name, font=header_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x_pos + (image_size - text_width) // 2
        text_y = y_pos + (method_header_height - text_height) // 2
        
        draw.text((text_x, text_y), method_name, fill="black", font=header_font)

    for seed_idx, seed in enumerate(tqdm(seeds, desc="Processing Seeds")):
        torch.manual_seed(seed)
        z = torch.randn(1, G.z_dim).to(device)
        w_source = G.mapping(z, None, truncation_psi=0.7)
        if len(w_source.shape) == 2:
            w_source = w_source.unsqueeze(1).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            original_img = G.synthesis(w_source, noise_mode='const')
        original_pil = tensor_to_pil(original_img).resize((image_size, image_size), Image.Resampling.LANCZOS)

        y_positions = []
        for edit_idx, edit_config in enumerate(edit_configs):
            y_pos = method_header_height + padding + edit_idx * (image_size + padding)
            y_positions.append(y_pos)
            
            label_x = figure_label_margin
            label_y = y_pos + image_size // 2
            
            bbox = draw.textbbox((0, 0), edit_config["description"], font=label_font)
            text_height = bbox[3] - bbox[1]
            draw.text((label_x, label_y - text_height // 2), edit_config["description"], 
                     fill="black", font=label_font)

            for method_idx, (method_name, method_fn) in enumerate(methods.items()):
                x_pos = figure_label_margin + edit_label_width + padding + method_idx * (image_size + padding)
                
                if method_name == "Original":
                    img_to_paste = original_pil
                else:
                    try:
                        w_edited, _ = method_fn(w_source.clone(), edit_config["edits"])
                        with torch.no_grad():
                            edited_img = G.synthesis(w_edited, noise_mode='const')
                        img_to_paste = tensor_to_pil(edited_img).resize((image_size, image_size), Image.Resampling.LANCZOS)
                    except Exception as e:
                        print(f"Error with {method_name}: {e}")
                        img_to_paste = original_pil 
                
                main_fig.paste(img_to_paste, (x_pos, y_pos))

        add_figure_labels(draw, grid_width, y_positions)

        figure_path = os.path.join(output_dir, f"figure_comparison_seed_{seed}.png")
        main_fig.save(figure_path, dpi=(300, 300))
        print(f" Saved research figure for seed {seed}: {figure_path}")

    print("\nGenerating individual method comparison figures...")
    
    for edit_config in edit_configs:
        fig_name = edit_config["name"].replace(" ", "_").lower()
        
        single_fig_width = num_methods * image_size + (num_methods + 1) * padding
        single_fig_height = method_header_height + image_size + 2 * padding
        
        single_fig = Image.new('RGB', (single_fig_width, single_fig_height), 'white')
        single_draw = ImageDraw.Draw(single_fig)
        
        for i, method_name in enumerate(method_names):
            x_pos = padding + i * (image_size + padding)
            y_pos = 10
            
            bbox = single_draw.textbbox((0, 0), method_name, font=header_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = x_pos + (image_size - text_width) // 2
            text_y = y_pos + (method_header_height - text_height) // 2
            
            single_draw.text((text_x, text_y), method_name, fill="black", font=header_font)

        torch.manual_seed(seeds[0])
        z = torch.randn(1, G.z_dim).to(device)
        w_source = G.mapping(z, None, truncation_psi=0.7)
        if len(w_source.shape) == 2:
            w_source = w_source.unsqueeze(1).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            original_img = G.synthesis(w_source, noise_mode='const')
        original_pil = tensor_to_pil(original_img).resize((image_size, image_size), Image.Resampling.LANCZOS)

        for method_idx, (method_name, method_fn) in enumerate(methods.items()):
            x_pos = padding + method_idx * (image_size + padding)
            y_pos = method_header_height + padding
            
            if method_name == "Original":
                img_to_paste = original_pil
            else:
                try:
                    w_edited, _ = method_fn(w_source.clone(), edit_config["edits"])
                    with torch.no_grad():
                        edited_img = G.synthesis(w_edited, noise_mode='const')
                    img_to_paste = tensor_to_pil(edited_img).resize((image_size, image_size), Image.Resampling.LANCZOS)
                except Exception as e:
                    print(f"Error with {method_name}: {e}")
                    img_to_paste = original_pil
            
            single_fig.paste(img_to_paste, (x_pos, y_pos))

        single_fig_path = os.path.join(output_dir, f"{fig_name}_comparison.png")
        single_fig.save(single_fig_path, dpi=(300, 300))
        print(f"Saved individual figure: {single_fig_path}")

    print("\n" + "="*80)
    print(" SUCCESSFUL")
    print(f"All figures saved in: {output_dir}/")
    print("Features:")
    print("- Publication-ready 300 DPI resolution")
    print("- Academic font sizes and styling")
    print("- Proper figure panel labels (a), (b), (c)")
    print("- Clear method comparisons")
    print("- Individual and comprehensive figures")
    print("="*80)

    print("\nGenerating special single seed figure...")
    # generate_single_seed_figure(G, clip_model, device, seed=15678)

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    G = load_stylegan2(DEVICE)
    clip_model = load_clip(DEVICE)

    generate_research_paper_figures(G, clip_model, DEVICE)