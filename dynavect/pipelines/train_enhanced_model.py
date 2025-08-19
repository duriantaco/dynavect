import lpips
import matplotlib.pyplot as plt

from trainer import Trainer
from direction_bank import GlobalDirectionBank
from modulator import DynamicContextualModulator

def train_enhanced_model(G, clip_model, device):
    use_w_plus = True
    direction_bank = GlobalDirectionBank(G, clip_model, device, use_w_plus)
    modulator = DynamicContextualModulator(
        w_dim=G.w_dim,
        num_layers=G.num_ws,
        use_w_plus=use_w_plus
    ).to(device)
    
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    train_edits = [
        {"neutral": "a man's face", "target": "a smiling man's face", "strength": 0.8},
        {"neutral": "a woman's face", "target": "a sad woman's face", "strength": 0.8},
        {"neutral": "a face", "target": "an angry face", "strength": 0.8},
        {"neutral": "a young man", "target": "an old man", "strength": 0.8},
        {"neutral": "a young woman", "target": "an old woman", "strength": 0.8},
        {"neutral": "an old man", "target": "a young man", "strength": 0.8},
        {"neutral": "an old woman", "target": "a young woman", "strength": 0.8},
        {"neutral": "a man with dark hair", "target": "a man with blonde hair", "strength": 1.2},
        {"neutral": "a woman with dark hair", "target": "a woman with blonde hair", "strength": 1.2},
        {"neutral": "an asian man with black hair", "target": "an asian man with blonde hair", "strength": 1.2},
        {"neutral": "a black woman with dark hair", "target": "a black woman with red hair", "strength": 1.2},
        {"neutral": "a man with blonde hair", "target": "a man with dark hair", "strength": 1.2},
        {"neutral": "a woman with blonde hair", "target": "a woman with brown hair", "strength": 1.2},
        {"neutral": "a man's face", "target": "a man's face with glasses", "strength": 1.0},
        {"neutral": "a woman's face", "target": "a woman's face with glasses", "strength": 1.0},
        {"neutral": "a man's face with glasses", "target": "a man's face without glasses", "strength": 1.0},
        {"neutral": "a woman's face with glasses", "target": "a woman's face without glasses", "strength": 1.0},
        {"neutral": "a woman with long hair", "target": "a woman with short hair", "strength": 1.0},
        {"neutral": "a man with short hair", "target": "a man with long hair", "strength": 1.0},
        {"neutral": "a white woman with straight hair", "target": "a white woman with curly hair", "strength": 0.8},
        {"neutral": "a black man with short hair", "target": "a black man with curly hair", "strength": 0.8},
        {"neutral": "a woman's face", "target": "a woman's face with makeup", "strength": 0.9},
        {"neutral": "a man's face", "target": "a man's face with makeup", "strength": 0.9},
        {"neutral": "a man's face", "target": "a tanned man's face", "strength": 0.7},
    ]

    val_edits = [
        {"neutral": "a woman's face", "target": "a surprised woman's face", "strength": 0.8},
        {"neutral": "a man's face without a beard", "target": "a man's face with a beard", "strength": 1.0},
        {"neutral": "a man", "target": "a professional man in a suit", "strength": 0.7},
        {"neutral": "a woman", "target": "a professional woman in a blouse", "strength": 0.7},
    ]
    
    trainer = Trainer(G, clip_model, modulator, direction_bank, device, lpips_fn)

    trainer.train(train_edits, steps=2000, val_edits=val_edits)

    history = trainer.training_history
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (key, values) in enumerate(history.items()):
        if idx < 4:
            axes[idx].plot(values)
            axes[idx].set_title(f'{key.capitalize()} Loss')
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel('Loss')
            axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")