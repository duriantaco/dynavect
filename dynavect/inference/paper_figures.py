import os
import torch

from fig_utils import load_stylegan2, load_clip, generate_grid_figures

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_stylegan2(device)
    clip_model = load_clip(device)

    # # smiling women
    # seeds = [42, 95, 101, 683, 909, 1138, 1492, 1776, 2001, 2077, 3000, 4126,
    #          4540, 8912, 9999, 1892, 1131, 4359, 12345, 11378, 15678]

    ## neutral men 
    seeds = [494, 202, 888, 4020, 27]

    ## limitations
    
    # seeds = [61, 121, 671, 993, 1121, 158, 1721, 2016, 2099, 3212, 4127,
    #          645, 313, 1562, 1567, 2526, 7272, 15121]

    # smiling women
    edit_configs = [
        {"name": "Age_Y2O", "description": "Young -> Old",
         "edits": [{"neutral": "a young person", "target": "an old person", "strength": 1.0}]},
        {"name": "Age_O2Y", "description": "Old -> Young",
         "edits": [{"neutral": "an old person", "target": "a young person", "strength": 1.0}]},
        {"name": "Hair", "description": "Dark -> Blonde Hair",
         "edits": [{"neutral": "a person with dark hair", "target": "a person with dyed blonde hair", "strength": 1.0}]},
        {"name": "Combined", "description": "Smile + Age + Hair",
         "edits": [
             {"neutral": "a face", "target": "an old woman with sunglasses", "strength": 1.4},
            {"neutral": "a face", "target": "a smiling face", "strength": 1.0},
            {"neutral": "a person with dark hair", "target": "a person with dyed blonde hair", "strength": 1.0}
         ]},
    ]
    
    # neutral man 
    # edit_configs = [
    #     {"name": "Sad_Face", "description": "Sad Face", "edits": [{"neutral": "a face", "target": "a sad face", "strength": 1.5}]},
    #     {"name": "Angry_Face", "description": "Angry Face", "edits": [{"neutral": "a face", "target": "an angry old face", "strength": 1.5}]},
    #     {"name": "Age_Y2O", "description": "Young -> Old", "edits": [{"neutral": "a young person", "target": "an old person", "strength": 1.0}]},
    #     {"name": "Age_O2Y", "description": "Old -> Young", "edits": [{"neutral": "an old person", "target": "a young person", "strength": 1.0}]},
    #     {"name": "Smile", "description": "Smile", "edits": [{"neutral": "a face", "target": "a smiling face", "strength": 1.0}]},

    # ]

# ## limitations
#     edit_configs = [
#         {"name": "Hair_Color", "description": "Dark → Blonde Hair", "edits": [{"neutral": "a person with dark hair", "target": "a person with dyed blonde hair", "strength": 1.2}]},
#         {"name": "Ethnicity_Change", "description": "Change to Asian", "edits": [{"neutral": "a caucasian face", "target": "an asian face", "strength": 1.2}]},
#     ]

    out_dir = "research_paper_figures_limits_neutral_men"
    os.makedirs(out_dir, exist_ok=True)

    generate_grid_figures(
        G, clip_model, device,
        seeds=seeds,
        edit_configs=edit_configs,
        out_dir=out_dir,
        checkpoint="enhanced_modulator_checkpoint_v1.pt",
        image_size=256,
    )
