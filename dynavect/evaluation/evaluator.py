# evaluation/evaluator.py
import os, json, cv2, clip, lpips, torch, numpy as np, torch.nn.functional as F
from clip_utils import clip_preprocess

try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHTFACE = True
except Exception:
    _HAS_INSIGHTFACE = False


def _to_bgr_uint8(img_tensor, max_side=1024):
    x = ((img_tensor[0].detach().clamp_(-1,1) + 1) * 0.5 * 255.0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    h, w = x.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / float(m)
        x = cv2.resize(x, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _has_cuda():
    try:
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except Exception:
        return False


class EditEvaluator:
    def __init__(self, device, prefer_pack="antelopev2"):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.face_apps = {}
        self._init_face(prefer_pack)

    def _init_face(self, prefer):
        if not _HAS_INSIGHTFACE:
            return
        packs = [prefer, "buffalo_l"] if prefer != "buffalo_l" else ["buffalo_l", "antelopev2"]
        providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if _has_cuda() else ['CPUExecutionProvider']
        for pack in packs:
            try:
                app = FaceAnalysis(name=pack, providers=providers)
                app.prepare(ctx_id=(0 if _has_cuda() else -1), det_size=(640,640), det_thresh=0.10)
                self.face_apps[pack] = app
            except Exception:
                pass

    def evaluate_edit(self, original_img, edited_img, target_text, clip_model, neutral_text="a face"):
        metrics = {}
        metrics['lpips'] = float(self.lpips_fn(original_img, edited_img).item())

        edited_resized = F.interpolate(edited_img, (224,224), mode='bilinear', align_corners=False)
        edited_normed  = clip_preprocess(edited_resized, self.device)
        with torch.no_grad():
            ef = F.normalize(clip_model.encode_image(edited_normed).float(), dim=-1)
            tf = F.normalize(clip_model.encode_text(clip.tokenize([target_text]).to(self.device)).float(), dim=-1)
            nf = F.normalize(clip_model.encode_text(clip.tokenize([neutral_text]).to(self.device)).float(), dim=-1)
        metrics['clip_score'] = float(F.cosine_similarity(ef, tf).item())
        metrics['clip_dir']   = float(F.cosine_similarity(ef, tf).item() - F.cosine_similarity(ef, nf).item())
        metrics['l2_distance']= float(F.mse_loss(original_img, edited_img).item())

        id_val, id_tag = self._identity(original_img, edited_img, clip_model)
        metrics['identity'] = float(id_val) if id_val is not None else float('nan')
        metrics['identity_metric'] = id_tag
        metrics['identity_detected'] = 1.0 if id_tag == "ArcFace" and not np.isnan(metrics['identity']) else 0.0
        return metrics

    def _identity(self, original_img, edited_img, clip_model):
        emb_o, emb_e = None, None
        if self.face_apps:
            o = self._prep_for_arcface(original_img)
            e = self._prep_for_arcface(edited_img)
            emb_o = self._arcface_embed(o)
            emb_e = self._arcface_embed(e)
            if emb_o is not None and emb_e is not None:
                return float((emb_o * emb_e).sum()), "ArcFace"
        with torch.no_grad():
            o = clip_preprocess(F.interpolate(original_img,(224,224),mode='bilinear',align_corners=False), self.device)
            e = clip_preprocess(F.interpolate(edited_img,(224,224),mode='bilinear',align_corners=False), self.device)
            fo = F.normalize(clip_model.encode_image(o).float(), dim=-1)
            fe = F.normalize(clip_model.encode_image(e).float(), dim=-1)
            v  = float(F.cosine_similarity(fo,fe).item())
            return v, "CLIP-proxy"

    def _prep_for_arcface(self, img_tensor):
        im = _to_bgr_uint8(img_tensor, max_side=1024)
        h, w = im.shape[:2]
        b = int(0.08 * max(h, w))
        if b > 0:
            im = cv2.copyMakeBorder(im, b, b, b, b, cv2.BORDER_CONSTANT, value=(127,127,127))
        return im

    def _arcface_embed(self, bgr_img):
        def pick_face(faces):
            if not faces: return None
            def area(b):
                x1,y1,x2,y2 = b
                return max(0,x2-x1)*max(0,y2-y1)
            f = max(faces, key=lambda o: (getattr(o,'det_score',0.0), area(o.bbox)))
            emb = getattr(f,'normed_embedding', None)
            if emb is None:
                raw = getattr(f,'embedding', None)
                if raw is None: return None
                raw = np.asarray(raw, dtype=np.float32)
                n = np.linalg.norm(raw) + 1e-8
                emb = raw / n
            return emb

        det_sizes = [(512,512),(640,640),(800,800),(1024,1024)]
        det_ths   = [0.05, 0.10, 0.15, 0.20]
        scales    = [1.0, 0.85, 1.25, 0.7, 1.5]

        for pack, app in self.face_apps.items():
            for ds in det_sizes:
                try: app.det_size = ds
                except Exception: pass
                for th in det_ths:
                    try: app.det_thresh = th
                    except Exception: pass
                    for s in scales:
                        h,w = bgr_img.shape[:2]
                        im = bgr_img if s==1.0 else cv2.resize(bgr_img,(int(w*s),int(h*s)),interpolation=cv2.INTER_LINEAR)
                        faces = app.get(im)
                        emb = pick_face(faces)
                        if emb is not None:
                            return emb
        return None

    def evaluate_method(self, method_fn, test_cases, G, clip_model):
        from collections import defaultdict
        allm = defaultdict(list)
        for case in test_cases:
            with torch.no_grad():
                z = torch.randn(1, G.z_dim, device=self.device)
                w = G.mapping(z, None, truncation_psi=0.7)
                if hasattr(method_fn,'use_w_plus') and getattr(method_fn,'use_w_plus',False):
                    if w.dim()==2: w = w.unsqueeze(1).repeat(1, G.num_ws, 1)
                img0 = G.synthesis(w, noise_mode='const')
                w1,_ = method_fn(w, case['edits'])
                img1 = G.synthesis(w1, noise_mode='const')
            tgt = ', '.join([e['target'] for e in case['edits']])
            neu = case['edits'][0].get('neutral','a face') if case.get('edits') else 'a face'
            m = self.evaluate_edit(img0, img1, tgt, clip_model, neutral_text=neu)
            for k,v in m.items(): allm[k].append(v)
        out = {}
        for k,v in allm.items():
            a = np.array(v, dtype=np.float64)
            out[k] = {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()), 'max': float(a.max())}
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model,_ = clip.load("ViT-B/32", device=device)
    from loaders import load_stylegan2
    G = load_stylegan2(device)

    torch.manual_seed(42)
    z = torch.randn(1, G.z_dim, device=device)
    w = G.mapping(z, None, truncation_psi=0.7)
    if w.dim()==2: w = w.unsqueeze(1).repeat(1, G.num_ws, 1)
    with torch.no_grad():
        img0 = G.synthesis(w, noise_mode='const')
        img1 = G.synthesis(w + 0.01*torch.randn_like(w), noise_mode='const')

    ev = EditEvaluator(device)
    out = ev.evaluate_edit(img0, img1, "a smiling face", clip_model, neutral_text="a face")
    print(json.dumps(out, indent=2))
