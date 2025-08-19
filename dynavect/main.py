import torch
from loaders import load_stylegan2, load_clip
from pipelines.train_enhanced_model import train_enhanced_model
from pipelines.inference import run_comprehensive_inference
from pipelines.ablation import run_ablation_study
from pipelines.optimizer_eval import run_optimizer_evaluation
from pipelines.comparison import create_comparison_figure

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    G = load_stylegan2(DEVICE)
    clip_model = load_clip(DEVICE)
    
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced DynaVect')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['train', 'inference', 'ablation', 'comparison', 'optimizer_eval', 'all'],
                       help='Execution mode')
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        train_enhanced_model(G, clip_model, DEVICE)
    
    if args.mode == 'inference' or args.mode == 'all':
        run_comprehensive_inference(G, clip_model, DEVICE)
    
    if args.mode == 'ablation' or args.mode == 'all':
        run_ablation_study(G, clip_model, DEVICE)
    
    if args.mode == 'optimizer_eval' or args.mode == 'all':
        run_optimizer_evaluation(G, clip_model, DEVICE, num_samples=20) 
    
    if args.mode == 'comparison' or args.mode == 'all':
        create_comparison_figure(G, clip_model, DEVICE)
    
    print("\nAll tasks completed.. Check enhanced_results/ folder for your results")