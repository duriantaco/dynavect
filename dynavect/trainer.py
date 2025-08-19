import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import lpips
from tqdm import tqdm
import clip
from clip_utils import clip_preprocess

class Trainer:
    def __init__(self, G, clip_model, modulator, direction_bank, device, lpips_loss_fn=None):
        self.G = G
        self.clip_model = clip_model
        self.modulator = modulator
        self.direction_bank = direction_bank
        self.device = device
        self.optimizer = Adam(modulator.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)
        self.scheduler = None
        self.lpips_loss_fn = lpips_loss_fn if lpips_loss_fn else lpips.LPIPS(net='alex').to(self.device)
        self.training_history = defaultdict(list)

    def _get_features(self, w_latent, text=None):
        with torch.no_grad():
            img = self.G.synthesis(w_latent, noise_mode='const')
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img_resized = clip_preprocess(img_resized, self.device)
            img_features = self.clip_model.encode_image(img_resized).float()
            img_features = F.normalize(img_features, dim=-1)

            text_features = None
            if text:
                tokens = clip.tokenize([text]).to(self.device)
                text_features = self.clip_model.encode_text(tokens).float()
                text_features = F.normalize(text_features, dim=-1)
            return img_features, text_features, img


    def train_step(self, edit_params):
        self.modulator.train()
        self.optimizer.zero_grad()
        
        w_source, w_target_gt = self.direction_bank.get_direction(
            edit_params["neutral"], 
            edit_params["target"],
            mode='optimize'
        )
        direction_gt = w_target_gt - w_source

        source_img_features, _, _ = self._get_features(w_source)
        _, target_text_features, _ = self._get_features(w_source, text=edit_params["target"])
            
        delta_predicted = self.modulator(source_img_features, target_text_features)
        
        direction_loss = F.mse_loss(delta_predicted, direction_gt)
        
        w_predicted = w_source + delta_predicted
        edited_img = self.G.synthesis(w_predicted, noise_mode='const')
        
        with torch.no_grad():
            edited_img_gt = self.G.synthesis(w_target_gt, noise_mode='const')

        reconstruction_loss = self.lpips_loss_fn(edited_img, edited_img_gt).mean()
        artifact_loss = delta_predicted.reshape(delta_predicted.shape[0], -1).norm(p=2, dim=1).mean() * 0.01
        
        total_loss = direction_loss + 0.5 * reconstruction_loss + artifact_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.modulator.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        losses = {
            'total': total_loss.item(),
            'direction': direction_loss.item(),
            'reconstruction': reconstruction_loss.item(),
            'artifact_reg': artifact_loss.item()
        }
        for k, v in losses.items():
            self.training_history[k].append(v)
        return losses

    def train(self, training_edits, steps=2000, val_edits=None):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=steps, eta_min=1e-6)
        print(f"\nTraining for {steps} steps...")
        pbar = tqdm(range(steps))
        
        for step in pbar:
            edit_params = training_edits[step % len(training_edits)]
            losses = self.train_step(edit_params)
            
            desc = f"Loss: {losses['total']:.4f} (Dir: {losses['direction']:.3f}, Recon: {losses['reconstruction']:.3f}, Art: {losses['artifact_reg']:.3f})"
            pbar.set_description(desc)
            
            if val_edits and step > 0 and step % 200 == 0:
                self.validate(val_edits)
        
        print("Training complete.")
        
        torch.save({
            'model_state_dict': self.modulator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': dict(self.training_history),
            'config': {
                'w_dim': self.modulator.w_dim,
                'num_layers': self.modulator.num_layers,
                'use_w_plus': self.modulator.use_w_plus
            }
        }, "enhanced_modulator_checkpoint_v1.pt")
        print("Checkpoint saved to enhanced_modulator_checkpoint_v1.pt")

    def validate(self, val_edits):
        self.modulator.eval()
        total_loss = 0
        with torch.no_grad():
            for edit_params in val_edits:
                z = torch.randn(1, self.G.z_dim).to(self.device)
                w_source = self.G.mapping(z, None, truncation_psi=0.7)
                if self.direction_bank.use_w_plus:
                    if len(w_source.shape) == 2:
                        w_source = w_source.unsqueeze(1).repeat(1, self.G.num_ws, 1)
                
                source_img_features, target_text_features, _ = self._get_features(w_source, edit_params["target"])
                
                delta_predicted = self.modulator(source_img_features, target_text_features)
                w_predicted = w_source + delta_predicted * edit_params.get("strength", 1.0)
                
                edited_img = self.G.synthesis(w_predicted, noise_mode='const')
                edited_resized = F.interpolate(edited_img, size=(224, 224), mode='bilinear', align_corners=False)
                edited_resized = clip_preprocess(edited_resized, self.device)
                edited_features = self.clip_model.encode_image(edited_resized).float()
                edited_features = F.normalize(edited_features, dim=-1)
                clip_loss = 1 - F.cosine_similarity(edited_features, target_text_features).mean()
                total_loss += clip_loss.item()
        
        print(f"\nStep - Validation Loss: {total_loss / len(val_edits):.4f}")
        return total_loss / len(val_edits)
