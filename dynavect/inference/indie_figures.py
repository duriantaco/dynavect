import os
import torch

from fig_utils import load_stylegan2, load_clip, generate_per_edit_figures

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_stylegan2(device)
    clip_model = load_clip(device)

    # seed = 15678
    seed = 888
    edit_configs = [
        {"name": "smile", "description": "Smile",
         "edits": [{"neutral": "a face", "target": "a smiling face", "strength": 1.0}]},
        {"name": "sad_face", "description": "Sad Face",
         "edits": [{"neutral": "a face", "target": "a sad face", "strength": 1.4}]},
        {"name": "angry_face", "description": "Angry Face",
         "edits": [{"neutral": "a face", "target": "an angry old face", "strength": 1.4}]},
        {"name": "age_young_to_old", "description": "Young → Old",
         "edits": [{"neutral": "a young person", "target": "an old person", "strength": 1.0}]},
        {"name": "age_old_to_young", "description": "Old → Young",
         "edits": [{"neutral": "an old person", "target": "a young person", "strength": 1.0}]},
        {"name": "hair_dark_to_blonde", "description": "Dark → Blonde Hair",
         "edits": [{"neutral": "a person with dark hair", "target": "a person with dyed blonde hair", "strength": 1.1}]},
        {"name": "combined_smile_age_hair", "description": "Smile + Age + Hair",
         "edits": [
             {"neutral": "a face", "target": "a smiling face", "strength": 1.0},
             {"neutral": "a young person", "target": "an old person", "strength": 1.0},
             {"neutral": "a person with dark hair", "target": "a person with dyed blonde hair", "strength": 1.1},
         ]},
    ]

    out_dir = "research_paper_figures"
    os.makedirs(out_dir, exist_ok=True)

    generate_per_edit_figures(
        G, clip_model, device,
        seed=seed,
        edit_configs=edit_configs,
        out_dir=out_dir,
        checkpoint="enhanced_modulator_checkpoint_v1.pt",
        image_size=256,
    )
