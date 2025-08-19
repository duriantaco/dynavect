import torch
import torch.nn.functional as F
import clip

from direction_bank import GlobalDirectionBank
from modulator import DynamicContextualModulator
from clip_utils import clip_preprocess

class DynaVect:
    def __init__(self, G, clip_model, device, use_w_plus=True):
        self.G = G
        self.clip_model = clip_model
        self.device = device
        self.use_w_plus = use_w_plus
        self.direction_bank = GlobalDirectionBank(G, clip_model, device, use_w_plus)
        self.modulator = DynamicContextualModulator(
            w_dim=G.w_dim, 
            num_layers=G.num_ws, 
            use_w_plus=use_w_plus
        ).to(device)
        
        self.disentanglement_bases = {}
        self._initialize_disentanglement_bases()

    def _initialize_disentanglement_bases(self):
        base_attributes = [
            ("identity", [("a face", "another face")]),
            ("gender", [("a woman's face", "a man's face")]),
            ("age", [("a young person", "an old person")]),
            ("expression", [("a neutral face", "a smiling face")]),
            ("race", [("an asian face", "a caucasian face"), ("a black face", "a caucasian face")])
        ]
        
        for attr_name, prompts in base_attributes:
            directions = []
            for neutral, target in prompts:
                dir_data = self.direction_bank.get_direction(neutral, target, mode='sample')
                directions.append(dir_data['base'])
            self.disentanglement_bases[attr_name] = directions

    def _get_image_features(self, w_latent):
        with torch.no_grad():
            img = self.G.synthesis(w_latent, noise_mode='const')
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img_resized = clip_preprocess(img_resized, self.device)
            img_features = self.clip_model.encode_image(img_resized).float()
            img_features = F.normalize(img_features, dim=-1)
            return img_features, img

    def _get_text_features(self, text):
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            text_features = self.clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            return text_features

    def _orthogonalize_multispace(self, direction, basis_vectors, preserve_attributes=None):
        if preserve_attributes is None:
            preserve_attributes = ['identity', 'gender']

        all_basis_sets = {'previous_edits': {'vectors': basis_vectors}}
        for attr_name in preserve_attributes:
            if attr_name in self.disentanglement_bases:
                all_basis_sets[attr_name] = {'vectors': self.disentanglement_bases[attr_name]}

        if self.use_w_plus and direction.dim() == 3:
            direction_flat = direction.reshape(direction.shape[0], -1)
        else:
            direction_flat = direction if direction.dim() == 2 else direction.reshape(1, -1)

        eps = 1e-8
        for _, basis_data in all_basis_sets.items():
            for basis in basis_data['vectors']:
                if basis.dim() == 2:
                    basis_flat = basis.unsqueeze(0).reshape(1, -1)
                else:
                    basis_flat = basis.reshape(1, -1)
                basis_unit = basis_flat / (basis_flat.norm(dim=-1, keepdim=True) + eps)
                dot = (direction_flat * basis_unit).sum(dim=-1, keepdim=True)
                direction_flat = direction_flat - dot * basis_unit

        if self.use_w_plus and direction.dim() == 3:
            direction = direction_flat.view_as(direction)
        else:
            direction = direction_flat if direction.dim() == 2 else direction_flat.view_as(direction)
        return direction


    def edit_combined(self, w_latent, edits, preserve_attributes=None):
        self.modulator.eval()
        w_current = w_latent.clone()

        all_deltas = []
        for edit_params in edits:
            neutral_text, target_text = edit_params["neutral"], edit_params["target"]
            strength = edit_params.get("strength", 1.0)
            modulation_strength = edit_params.get("modulation_strength", 0.7)

            direction_data = self.direction_bank.get_direction(neutral_text, target_text, mode='sample')
            baseline_direction = direction_data['base']

            source_img_features, _ = self._get_image_features(w_latent) 
            target_text_features = self._get_text_features(target_text)
            
            with torch.no_grad():
                predicted_delta = self.modulator(source_img_features, target_text_features)

            final_direction = baseline_direction + predicted_delta * modulation_strength
            all_deltas.append(final_direction * strength)

        if not all_deltas:
            return w_current, []
            
        combined_delta = torch.sum(torch.stack(all_deltas), dim=0)

        if preserve_attributes:
            print("Applying orthogonalization to the final combined direction.")
            combined_delta = self._orthogonalize_multispace(
                combined_delta, 
                [], 
                preserve_attributes=preserve_attributes
            )
            
        w_edited = w_latent + combined_delta
        
        edit_history = [{'direction': combined_delta, 'edit_type': 'combined'}]
        
        return w_edited, edit_history