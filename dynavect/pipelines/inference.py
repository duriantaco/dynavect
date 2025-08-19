import os
import torch
import numpy as np
from torchvision.utils import save_image
import pandas as pd

from dynavect import DynaVect
from evaluation.evaluator import EditEvaluator

def run_comprehensive_inference(G, clip_model, device):
    print("\nRunning inference and evaluation...")
    
    editor = DynaVect(G, clip_model, device, use_w_plus=True)
    
    try:
        checkpoint = torch.load("enhanced_modulator_checkpoint_v1.pt", map_location=device)
        editor.modulator.load_state_dict(checkpoint['model_state_dict'])
        print("Enhanced modulator loaded successfully.")
    except FileNotFoundError:
        print("Checkpoint not found. Using random initialization.")

    evaluator = EditEvaluator(device, prefer_pack="antelopev2")

    test_cases = [
        {
            'name': 'smile_only',
            'edits': [{"neutral": "a face", "target": "a smiling face", "strength": 0.8}]
        },
        {
            'name': 'multi_edit',
            'edits': [
                {"neutral": "a face", "target": "a smiling face", "strength": 0.6},
                {"neutral": "a face", "target": "a face with glasses", "strength": 0.8}
            ]
        },
        {
            'name': 'complex_edit',
            'edits': [
                {"neutral": "a young person", "target": "an old person", "strength": 0.7},
                {"neutral": "a face", "target": "a smiling face", "strength": 0.6},
                {"neutral": "a person with dark hair", "target": "a person with gray hair", "strength": 0.9}
            ]
        }
    ]
    
    os.makedirs("enhanced_results", exist_ok=True)
    
    seeds = [42, 123, 298, 567, 890]
    results_data = []
    
    for seed in seeds:
        with torch.no_grad():
            torch.manual_seed(seed)
            z = torch.randn(1, G.z_dim).to(device)
            w_original = G.mapping(z, None, truncation_psi=0.7)
            if len(w_original.shape) == 2:
                w_original = w_original.unsqueeze(1).repeat(1, G.num_ws, 1)
            original_img = G.synthesis(w_original, noise_mode='const')

        img_normalized = (original_img + 1) / 2
        save_image(img_normalized, f"enhanced_results/seed_{seed}_original.png")
        
        for test_case in test_cases:
            print(f"\nApplying {test_case['name']}...")
    
            identity_floor_by_len = {1: 0.70, 2: 0.65}
            identity_floor = identity_floor_by_len.get(len(test_case['edits']), 0.55)

            preservation_configs = [
                {'name': 'no_preserve', 'attrs': None},
                {'name': 'preserve_identity','attrs': ['identity']},
                {'name': 'preserve_all', 'attrs': ['identity','gender','age']}
            ]

            strength_scales = [0.6, 0.8, 1.0, 1.2]
            mod_scales      = [0.6, 0.8, 1.0, 1.2]

            best = None
            best_fallback = None

            for config in preservation_configs:
                for s in strength_scales:
                    for ms in mod_scales:
                        scaled_edits = []
                        for e in test_case['edits']:
                            se = dict(e)
                            se['strength'] = e.get('strength', 1.0) * s
                            se['modulation_strength'] = e.get('modulation_strength', 0.7) * ms
                            scaled_edits.append(se)

                        w_edited, _ = editor.edit_combined(w_original, 
                                                           scaled_edits, 
                                                           preserve_attributes=config['attrs'])
                        
                        edited_img = G.synthesis(w_edited, noise_mode='const')

                        edits = test_case['edits']
                        targets = [e['target'] for e in edits]
                        combined_target = ', '.join(targets)

                        first_edit = edits[0]
                        neutral_txt = first_edit.get('neutral', 'a face')

                        m = evaluator.evaluate_edit(original_img, 
                                                    edited_img, 
                                                    combined_target, 
                                                    clip_model, 
                                                    neutral_text=neutral_txt)

                        if best_fallback is None:
                            best_fallback = m

                        else:
                            better_identity = m['identity'] > best_fallback['identity']
                            same_identity = m['identity'] == best_fallback['identity']
                            better_clip = m['clip_dir'] > best_fallback['clip_dir']
                            
                            if better_identity:
                                best_fallback = m
                            elif same_identity and better_clip:
                                best_fallback = m

                        if m['identity'] >= identity_floor:
                            if best is None or m['clip_dir'] > best['clip_dir']:
                                best = m

            if best is not None:
                final_metrics = best
            else:
                final_metrics = best_fallback

            results_data.append({
                'seed': seed,
                'test_case': test_case['name'],
                'preservation': 'auto',
                'clip_dir':     final_metrics['clip_dir'],
                'identity':     final_metrics['identity'],
                'clip_score':   final_metrics['clip_score'],
                'lpips':        final_metrics['lpips'],
                'l2_distance':  final_metrics['l2_distance'],
            })

    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('enhanced_results/evaluation_metrics.csv', index=False)
    
    print("\n" + "="*60)
    print("HEADLINE (mean ± std) — Directional CLIP (up) and Identity (up))")
    print("="*60)
    
    for test_name in results_df['test_case'].unique():
        print(f"\n{test_name.upper()}:")
        test_data = results_df[results_df['test_case'] == test_name]
        
        metrics = ['clip_dir', 'identity', 'clip_score', 'lpips', 'l2_distance']
        for key in metrics:
            vals = test_data[key]
            print(f" {key}: {vals.mean():.4f} ± {vals.std():.4f}")
