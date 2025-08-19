import torch
import torch.nn.functional as F
import lpips
import clip
from tqdm import tqdm
from clip_utils import clip_preprocess

class GlobalDirectionBank:
    def __init__(self, G, clip_model, device, use_w_plus=True):
        self.G = G
        self.clip_model = clip_model
        self.device = device
        self.use_w_plus = use_w_plus
        self.w_dim = G.w_dim
        self.num_layers = G.num_ws
        self.directions_cache = {}

    def get_direction(self, neutral_text, target_text, mode='sample'):
        if mode == 'sample':
            cache_key = (neutral_text, target_text, 'sample')
            if cache_key in self.directions_cache:
                return self.directions_cache[cache_key]
            
            print(f"Sampling global direction: '{neutral_text}' -> '{target_text}'")
            direction = self._find_direction_sampling(neutral_text, target_text, num_samples=2000)
            self.directions_cache[cache_key] = {'base': direction}
            return self.directions_cache[cache_key]

        elif mode == 'optimize':
            print(f"Generating training pair for: '{target_text}'")
            w_start, w_target = self._generate_optimized_pair(target_text, optimization_steps=100)
            return w_start, w_target
        else:
            raise ValueError("Either 'sample' or 'optimize'")

    def _find_direction_sampling(self, neutral_text, target_text, num_samples):
        with torch.no_grad():
            neutral_tokens = clip.tokenize([neutral_text]).to(self.device)
            target_tokens = clip.tokenize([target_text]).to(self.device)

            neutral_features = self.clip_model.encode_text(neutral_tokens).float()
            target_features = self.clip_model.encode_text(target_tokens).float()
            neutral_features = F.normalize(neutral_features, dim=-1)
            target_features  = F.normalize(target_features,  dim=-1)

            z = torch.randn(num_samples, self.G.z_dim).to(self.device)
            w = self.G.mapping(z, None, truncation_psi=0.7)

            if self.use_w_plus and len(w.shape) == 2:
                w = w.unsqueeze(1).repeat(1, self.num_layers, 1)

            batch_size = 20
            neutral_scores, target_scores = [], []

            for i in tqdm(range(0, num_samples, batch_size), desc="Scoring faces for global dir"):
                start = i
                end = i + batch_size
                batch_w = w[start:end]

                imgs = self.G.synthesis(batch_w, noise_mode='const')
                imgs_resized = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
                imgs_resized = clip_preprocess(imgs_resized, self.device)
                img_features = self.clip_model.encode_image(imgs_resized).float()
                img_features = F.normalize(img_features, dim=-1)

                neutral_scores.append((img_features * neutral_features).sum(dim=-1))
                target_scores.append((img_features * target_features).sum(dim=-1))

        neutral_scores = torch.cat(neutral_scores)
        target_scores  = torch.cat(target_scores)

        top_k = 20
        neutral_indices = torch.topk(neutral_scores, top_k).indices
        target_indices  = torch.topk(target_scores,  top_k).indices

        # target_indices = target_indices[~torch.isin(target_indices, neutral_indices)]

        target_mean  = w[target_indices].mean(dim=0)
        neutral_mean = w[neutral_indices].mean(dim=0)
        direction = target_mean - neutral_mean
        return direction

    def _generate_optimized_pair(self, target_text, optimization_steps):

        target_tokens = clip.tokenize([target_text]).to(self.device)
        with torch.no_grad():
            target_features = self.clip_model.encode_text(target_tokens).float()
            target_features = F.normalize(target_features, dim=-1)

        z = torch.randn(1, self.G.z_dim, device=self.device)
        w_start = self.G.mapping(z, None, truncation_psi=0.7)
        if self.use_w_plus and w_start.dim() == 2:
            w_start = w_start.unsqueeze(1).repeat(1, self.num_layers, 1)

        w_opt = w_start.clone().detach().requires_grad_(True)

        lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        optimizer = torch.optim.Adam([w_opt], lr=0.01)

        with torch.no_grad():
            img_start = self.G.synthesis(w_start, noise_mode='const')

        for _ in range(optimization_steps):
            optimizer.zero_grad()
            img_edited = self.G.synthesis(w_opt, noise_mode='const')
            img_resized = F.interpolate(img_edited, size=(224, 224), mode='bilinear', align_corners=False)
            img_resized = clip_preprocess(img_resized, self.device)
            img_features = self.clip_model.encode_image(img_resized).float()
            img_features = F.normalize(img_features, dim=-1)
            clip_loss = 1 - (img_features * target_features).sum(dim=-1).mean()

            identity_loss = lpips_loss_fn(img_start, img_edited).mean()
            latent_loss = F.mse_loss(w_start, w_opt)

            total_loss = clip_loss + 0.8 * identity_loss + 0.5 * latent_loss
            total_loss.backward()
            optimizer.step()

        return w_start.detach(), w_opt.detach()
