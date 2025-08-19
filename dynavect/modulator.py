import torch
import torch.nn as nn

class DynamicContextualModulator(nn.Module):
    """
    simple light weight modulator that takes in img features and text features and outputs a delta
    to be added to the w_latent vector
    """
    def __init__(self, w_dim=512, clip_dim=512, num_layers=18, use_w_plus=True):
        super().__init__()
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.use_w_plus = use_w_plus
        

        self.net = nn.Sequential(
            nn.Linear(clip_dim * 2, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
        )
        
        if use_w_plus:
            self.layer_heads = nn.ModuleList([
                nn.Linear(1024, w_dim) for _ in range(num_layers)
            ])
        else:
            self.output_head = nn.Linear(1024, w_dim)

        self.layer_attention = nn.Sequential(
            nn.Linear(1024, num_layers),
            nn.Softmax(dim=-1)
        )

    def forward(self, source_img_features, target_text_features):
        context = torch.cat([source_img_features, target_text_features], dim=-1)
        features = self.net(context)

        if self.use_w_plus and len(features.shape) == 2:
            layer_deltas = []
            for layer_head in self.layer_heads:
                delta = layer_head(features)
                layer_deltas.append(delta)

            delta = torch.stack(layer_deltas, dim=1)
            
            attention_weights = self.layer_attention(features).unsqueeze(-1)
            delta = delta * attention_weights
        else:
            delta = self.output_head(features)
        
        return delta
