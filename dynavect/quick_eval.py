import json, torch, clip
from evaluation.evaluator import EditEvaluator

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)

    H = W = 512
    img0 = (torch.rand(1, 3, H, W, device=device) * 2) - 1
    img1 = (img0 + 0.02 * torch.randn_like(img0)).clamp(-1, 1)

    evaluator = EditEvaluator(device)
    metrics = evaluator.evaluate_edit(img0, img1, "a smiling face", clip_model)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
