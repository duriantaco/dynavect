import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import clip
import lpips
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import sys

from dynavect.dynavect import DynaVect
from dynavect.direction_bank import GlobalDirectionBank
from dynavect.clip_utils import clip_preprocess

from pathlib import Path

_here = Path(__file__).resolve()
CANDIDATES = [
    _here.parent / 'stylegan2-ada-pytorch', 
    _here.parents[1] / 'stylegan2-ada-pytorch', 
    _here.parents[2] / 'stylegan2-ada-pytorch',
    Path(os.environ.get('STYLEGAN2_ADA_DIR', '')),
]
for p in CANDIDATES:
    if p and str(p) != '' and p.exists():
        sys.path.insert(0, str(p.resolve()))
        break
else:
    raise RuntimeError(
        "Could not locate stylegan2-ada-pytorch. "
        "Set STYLEGAN2_ADA_DIR or place the repo alongside your project."
    )

import clip
import dnnlib, legacy

def load_stylegan2(device):
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    with dnnlib.util.open_url(url) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G


def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model


def tensor_to_pil(t):
    t = (t.clone().detach().squeeze(0).permute(1, 2, 0) + 1.) / 2.
    t = (t * 255).clip(0, 255).to(torch.uint8)
    return Image.fromarray(t.cpu().numpy(), 'RGB')


def get_font(size, bold=False):
    paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        try:
            if bold and "arial.ttf" in p.lower():
                bp = p.replace("arial.ttf", "arialbd.ttf")
                if os.path.exists(bp):
                    return ImageFont.truetype(bp, size)
            return ImageFont.truetype(p, size)
        except:
            continue
    return ImageFont.load_default()


def add_panel_labels(draw, y_positions, font_size=16):
    font = get_font(font_size)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for i, y in enumerate(y_positions):
        if i < len(labels):
            draw.text((10, y - 30), labels[i], fill="black", font=font)


class StyleCLIPOptimizer:
    def __init__(self, G, clip_model, device, steps=100):
        self.G = G
        self.clip_model = clip_model
        self.device = device
        self.steps = steps
        self.lpips = lpips.LPIPS(net='alex').to(device)
        self.use_w_plus = True

    def __call__(self, w_latent, edits):
        with torch.no_grad():
            img_orig = self.G.synthesis(w_latent, noise_mode='const')
        target = ", ".join([e['target'] for e in edits])
        return self._optimize(w_latent, img_orig, target), None

    def _optimize(self, w_start, img_orig, text):
        tok = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            tfeat = self.clip_model.encode_text(tok).float()
            tfeat = F.normalize(tfeat, dim=-1)

        w_opt = w_start.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w_opt], lr=0.01)

        for _ in tqdm(range(self.steps), desc=f"Optimizing '{text}'"):
            opt.zero_grad()
            img = self.G.synthesis(w_opt, noise_mode='const')
            x = F.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
            x = clip_preprocess(x, self.device)
            if x.dtype != torch.float32:
                x = x.float()
            if tfeat.dtype != torch.float32:
                tfeat = tfeat.float()
            if x.device != self.device:
                x = x.to(self.device)
            if tfeat.device != self.device:
                tfeat = tfeat.to(self.device)
            imfeat = self.clip_model.encode_image(x).float()
            imfeat = F.normalize(imfeat, dim=-1)
            clip_loss = 1 - (imfeat * tfeat).sum(dim=-1).mean()
            id_loss = self.lpips(img_orig, img).mean()
            lat_loss = F.mse_loss(w_start, w_opt)
            (clip_loss + 0.8 * id_loss + 0.5 * lat_loss).backward()
            opt.step()
        return w_opt.detach()


class GlobalDirectionOnly:
    def __init__(self, G, clip_model, device):
        self.bank = GlobalDirectionBank(G, clip_model, device)
        self.use_w_plus = self.bank.use_w_plus

    def __call__(self, w_latent, edits):
        w = w_latent.clone()
        for e in edits:
            d = self.bank.get_direction(e['neutral'], e['target'], mode='sample')['base']
            w = w + d * e.get('strength', 1.0)
        return w, None


def build_methods(G, clip_model, device, checkpoint="enhanced_modulator_checkpoint_v1.pt",
                  preserve_attrs=('identity',)):
    methods = OrderedDict()
    methods["Original"] = None

    editor = DynaVect(G, clip_model, device)
    try:
        ckpt = torch.load(checkpoint, map_location=device)
        editor.modulator.load_state_dict(ckpt['model_state_dict'])
    except FileNotFoundError:
        pass
    methods["DynaVect (Ours)"] = lambda w, e: editor.edit_combined(w, e, preserve_attributes=list(preserve_attrs))

    methods["StyleCLIP-O"] = StyleCLIPOptimizer(G, clip_model, device)
    methods["StyleCLIP-G"] = GlobalDirectionOnly(G, clip_model, device)
    return methods


def generate_single_seed_figure(G, clip_model, device, seed, edit_config, out_dir,
                                checkpoint="enhanced_modulator_checkpoint_v1.pt",
                                image_size=300, pad=25, header_h=80, label_h=50):
    os.makedirs(out_dir, exist_ok=True)
    methods = build_methods(G, clip_model, device, checkpoint)
    names = list(methods.keys())

    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim).to(device)
    w = G.mapping(z, None, truncation_psi=0.7)
    if w.dim() == 2:
        w = w.unsqueeze(1).repeat(1, G.num_ws, 1)

    with torch.no_grad():
        img0 = G.synthesis(w, noise_mode='const')
    pil0 = tensor_to_pil(img0).resize((image_size, image_size), Image.Resampling.LANCZOS)

    imgs = [pil0]
    for n in names[1:]:
        try:
            w_ed, _ = methods[n](w.clone(), edit_config["edits"])
            with torch.no_grad():
                im = G.synthesis(w_ed, noise_mode='const')
            imgs.append(tensor_to_pil(im).resize((image_size, image_size), Image.Resampling.LANCZOS))
        except:
            imgs.append(pil0)

    W = len(names) * image_size + (len(names) + 1) * pad
    H = header_h + image_size + label_h + 2 * pad
    fig = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(fig)
    prompt_font = get_font(24, bold=True)
    label_font = get_font(18)

    ptxt = edit_config.get("prompt", "")
    bbox = draw.textbbox((0, 0), ptxt, font=prompt_font)
    draw.text(((W - (bbox[2]-bbox[0])) // 2, (header_h - (bbox[3]-bbox[1])) // 2),
              ptxt, fill="black", font=prompt_font)

    for i, (n, pil) in enumerate(zip(names, imgs)):
        x = pad + i * (image_size + pad)
        y = header_h + pad
        fig.paste(pil, (x, y))
        bb = draw.textbbox((0, 0), n, font=label_font)
        draw.text((x + (image_size - (bb[2]-bb[0])) // 2, y + image_size + 10),
                  n, fill="black", font=label_font)

    path = os.path.join(out_dir, f"single_seed_{seed}_comparison.png")
    fig.save(path, dpi=(300, 300))
    return path


def generate_grid_figures(G, clip_model, device, seeds, edit_configs, out_dir,
                          checkpoint="enhanced_modulator_checkpoint_v1.pt",
                          image_size=256, pad=20, header_h=80, edit_label_w=200, left_margin=50):
    os.makedirs(out_dir, exist_ok=True)
    methods = build_methods(G, clip_model, device, checkpoint)
    names = list(methods.keys())

    for seed in tqdm(seeds, desc="Processing Seeds"):
        torch.manual_seed(seed)
        z = torch.randn(1, G.z_dim).to(device)
        w = G.mapping(z, None, truncation_psi=0.7)
        if w.dim() == 2:
            w = w.unsqueeze(1).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            img0 = G.synthesis(w, noise_mode='const')
        pil0 = tensor_to_pil(img0).resize((image_size, image_size), Image.Resampling.LANCZOS)

        W = left_margin + edit_label_w + len(names) * image_size + (len(names) + 1) * pad
        H = header_h + len(edit_configs) * (image_size + pad) + pad
        fig = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(fig)

        header_font = get_font(16)
        label_font = get_font(14)

        for i, n in enumerate(names):
            x = left_margin + edit_label_w + pad + i * (image_size + pad)
            y = 10
            bb = draw.textbbox((0, 0), n, font=header_font)
            draw.text((x + (image_size - (bb[2]-bb[0])) // 2, y + (header_h - (bb[3]-bb[1])) // 2),
                      n, fill="black", font=header_font)

        y_positions = []
        for r, cfg in enumerate(edit_configs):
            y = header_h + pad + r * (image_size + pad)
            y_positions.append(y)
            lx = left_margin
            ly = y + image_size // 2
            bb = draw.textbbox((0, 0), cfg["description"], font=label_font)
            draw.text((lx, ly - (bb[3]-bb[1]) // 2), cfg["description"], fill="black", font=label_font)

            for c, (n, fn) in enumerate(methods.items()):
                x = left_margin + edit_label_w + pad + c * (image_size + pad)
                if n == "Original":
                    fig.paste(pil0, (x, y))
                else:
                    try:
                        w_ed, _ = fn(w.clone(), cfg["edits"])
                        with torch.no_grad():
                            im = G.synthesis(w_ed, noise_mode='const')
                        pil = tensor_to_pil(im).resize((image_size, image_size), Image.Resampling.LANCZOS)
                    except:
                        pil = pil0
                    fig.paste(pil, (x, y))

        add_panel_labels(draw, y_positions)
        path = os.path.join(out_dir, f"figure_comparison_seed_{seed}.png")
        fig.save(path, dpi=(300, 300))
    
def generate_per_edit_figures(G, clip_model, device, seed, edit_configs, out_dir,
                              checkpoint="enhanced_modulator_checkpoint_v1.pt",
                              image_size=256, pad=20, header_h=80):
    os.makedirs(out_dir, exist_ok=True)
    methods = build_methods(G, clip_model, device, checkpoint)
    names = list(methods.keys())

    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim).to(device)
    w = G.mapping(z, None, truncation_psi=0.7)
    if w.dim() == 2:
        w = w.unsqueeze(1).repeat(1, G.num_ws, 1)

    with torch.no_grad():
        img0 = G.synthesis(w, noise_mode='const')
    pil0 = tensor_to_pil(img0).resize((image_size, image_size), Image.Resampling.LANCZOS)

    header_font = get_font(16)

    for cfg in edit_configs:
        fig_w = len(names) * image_size + (len(names) + 1) * pad
        fig_h = header_h + image_size + 2 * pad
        fig = Image.new('RGB', (fig_w, fig_h), 'white')
        draw = ImageDraw.Draw(fig)

        for i, n in enumerate(names):
            x = pad + i * (image_size + pad)
            y = 10
            bb = draw.textbbox((0, 0), n, font=header_font)
            draw.text((x + (image_size - (bb[2]-bb[0])) // 2, y + (header_h - (bb[3]-bb[1])) // 2),
                      n, fill="black", font=header_font)

        for i, (n, fn) in enumerate(methods.items()):
            x = pad + i * (image_size + pad)
            y = header_h + pad
            if n == "Original":
                pil = pil0
            else:
                try:
                    w_ed, _ = fn(w.clone(), cfg["edits"])
                    with torch.no_grad():
                        im = G.synthesis(w_ed, noise_mode='const')
                    pil = tensor_to_pil(im).resize((image_size, image_size), Image.Resampling.LANCZOS)
                except:
                    pil = pil0
            fig.paste(pil, (x, y))

        name = cfg.get("name", cfg.get("description", "edit")).replace(" ", "_").lower()
        path = os.path.join(out_dir, f"{name}_comparison.png")
        fig.save(path, dpi=(300, 300))

