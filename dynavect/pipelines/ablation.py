import torch
import json

from direction_bank import GlobalDirectionBank
from dynavect import DynaVect
from evaluation.evaluator import EditEvaluator

def run_ablation_study(G, clip_model, device):
    print("Running ablation...")
    
    use_w_plus = True
    direction_bank = GlobalDirectionBank(G, clip_model, device, use_w_plus)
    
    class GlobalOnly:
        def __init__(self, direction_bank):
            self.direction_bank = direction_bank
            self.use_w_plus = direction_bank.use_w_plus
            
        def __call__(self, w_latent, edits):
            w_current = w_latent.clone()
            for edit in edits:
                direction_data = self.direction_bank.get_direction(
                    edit['neutral'], edit['target']
                )
                w_current += direction_data['base'] * edit.get('strength', 1.0)
            return w_current, None

    class GlobalPlusOrtho:
        def __init__(self, direction_bank):
            self.direction_bank = direction_bank
            self.use_w_plus = direction_bank.use_w_plus
            
        def _orthogonalize(self, direction, basis_vectors):
            for basis in basis_vectors:
                proj = (torch.dot(direction.flatten(), basis.flatten()) / 
                       torch.dot(basis.flatten(), basis.flatten())) * basis
                direction = direction - proj
            return direction
            
        def __call__(self, w_latent, edits):
            w_current = w_latent.clone()
            applied_directions = []
            
            for edit in edits:
                direction_data = self.direction_bank.get_direction(
                    edit['neutral'], edit['target']
                )
                direction = direction_data['base']
                
                if applied_directions:
                    direction = self._orthogonalize(direction, applied_directions)
                
                w_current += direction * edit.get('strength', 1.0)
                applied_directions.append(direction)
                
            return w_current, None
    
    full_dynavect = DynaVect(G, clip_model, device, use_w_plus)
    
    try:
        checkpoint = torch.load("enhanced_modulator_checkpoint_v1.pt", map_location=device)
        full_dynavect.modulator.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint")
    except FileNotFoundError:
        print("No checkpoint found, using default weights")
    
    ablation_test_cases = []
    # for i in range(20):
    for _ in range(20):
        edits = [
            {"neutral": "a face", "target": "a smiling face", "strength": 0.9},
            {"neutral": "a young person", "target": "an old person", "strength": 0.7}
        ]
        ablation_test_cases.append({'edits': edits})
    
    evaluator = EditEvaluator(device, prefer_pack="buffalo_l")
    methods = {
        'global_only': GlobalOnly(direction_bank),
        'global_ortho': GlobalPlusOrtho(direction_bank),
        'full_dynavect': lambda w, e: full_dynavect.edit_combined(w, e, ['identity', 'gender'])
    }
    
    ablation_results = {}
    for method_name, method_fn in methods.items():
        print(f"\nEvaluating {method_name}...")
        results = evaluator.evaluate_method(method_fn, ablation_test_cases, G, clip_model)
        ablation_results[method_name] = results
    
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print(f"{'Method':<20} {'ΔCLIP':<20} {'Identity':<20} {'LPIPS':<20}")
    print("-"*70)
    
    for method_name, results in ablation_results.items():
        clip_dir = f"{results['clip_dir']['mean']:.4f} ± {results['clip_dir']['std']:.4f}"
        identity = f"{results['identity']['mean']:.4f} ± {results['identity']['std']:.4f}"
        lpips_score = f"{results['lpips']['mean']:.4f} ± {results['lpips']['std']:.4f}"
        print(f"{method_name:<20} {clip_dir:<20} {identity:<20} {lpips_score:<20}")
    
    import os
    os.makedirs('enhanced_results', exist_ok=True)
    with open('enhanced_results/ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print("\n Ablation results saved to enhanced_results/ablation_study.json")
