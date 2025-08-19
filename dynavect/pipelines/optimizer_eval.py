import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from direction_bank import GlobalDirectionBank
from evaluation.evaluator import EditEvaluator

def run_optimizer_evaluation(G, clip_model, device, num_samples=10):
    print("\n" + "="*50)
    print("Running eval for StyleCLIP-o Baseline")
    print("="*50)

    direction_bank = GlobalDirectionBank(G, clip_model, device, use_w_plus=True)
    evaluator = EditEvaluator(device, prefer_pack="buffalo_l")
    
    target_text = "a smiling face, an old person, a person with blonde hair"
    
    all_metrics = defaultdict(list)

    for i in tqdm(range(num_samples), desc="Evaluating Optimizer Baseline"):
        w_source, w_edited = direction_bank._generate_optimized_pair(target_text, optimization_steps=100)
        
        with torch.no_grad():
            original_img = G.synthesis(w_source, noise_mode='const')
            edited_img = G.synthesis(w_edited, noise_mode='const')

        metrics = evaluator.evaluate_edit(original_img, edited_img, target_text, clip_model, neutral_text="a face")
        
        for k, v in metrics.items():
            all_metrics[k].append(v)
            
    print("\n--- Optimizer Baseline Results ---")
    for metric in ['clip_dir', 'identity']:
        if metric in all_metrics:
            mean = np.mean(all_metrics[metric])
            std  = np.std(all_metrics[metric])
            name = "Directional CLIP (ΔCLIP)" if metric=='clip_dir' else "Identity"
            print(f"  {name}: {mean:.4f} ± {std:.4f}")

    print("\nSecondary:")
    for metric in ['clip_score', 'lpips', 'l2_distance']:
        if metric in all_metrics:
            mean = np.mean(all_metrics[metric])
            std  = np.std(all_metrics[metric])
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
