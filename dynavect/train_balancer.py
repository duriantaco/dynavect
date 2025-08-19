from __future__ import annotations
import torch
import torch.nn as nn
import clip
from torch.optim import Adam

class AdaptiveLossBalancer(nn.Module):
    def __init__(self, clip_dim=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, neutral_features, target_features):
        x = torch.cat([neutral_features, target_features], dim=-1)
        return self.net(x)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    training_data = [
        (("a face", "a smiling face"), 0.9),
        (("a face", "a sad face"), 0.85),
        (("a young person", "an old person"), 0.6),
        (("a face", "a face with a sharper nose"), 0.5), 
        (("a face", "a face with a sharper jaw"), 0.4),
        (("a face", "a face with sunglasses"), 0.2),
        (("a face", "a face with earrings"), 0.7),
        (("a white woman", "an asian woman"), 0.05),
    ]

    balancer = AdaptiveLossBalancer().to(device)
    optimizer = Adam(balancer.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Training AdaptiveLossBalancer...")
    for epoch in range(200):
        total_loss = 0
        for (neutral_text, target_text), target_weight in training_data:
            optimizer.zero_grad()
            
            with torch.no_grad():
                neutral_tokens = clip.tokenize([neutral_text]).to(device)
                target_tokens = clip.tokenize([target_text]).to(device)
                neutral_features = clip_model.encode_text(neutral_tokens).float()
                target_features = clip_model.encode_text(target_tokens).float()
                neutral_features = torch.nn.functional.normalize(neutral_features, dim=-1)
                target_features  = torch.nn.functional.normalize(target_features,  dim=-1) 

            pred_weight = balancer(neutral_features, target_features)
            target_tensor = torch.tensor(target_weight)
            target_tensor = target_tensor.to(device)
            loss = loss_fn(pred_weight.squeeze(), target_tensor)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(training_data):.6f}")

    torch.save(balancer.state_dict(), "loss_balancer.pt")
    print("\n AdaptiveLossBalancer trained and saved to loss_balancer.pt")