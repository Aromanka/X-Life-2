import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import math
from PIL import Image
import numpy as np
from hf_mirror_config import with_hf_mirror, enable_hf_mirror

# Global enable HF mirror
enable_hf_mirror()

class SimpleLanguageModel(nn.Module):
    """Simple language model as backup for pre-trained models"""

    def __init__(self, vocab_size=21128, hidden_size=768, num_layers=6, num_heads=12):
        super(SimpleLanguageModel, self).__init__()

        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'vocab_size': vocab_size
        })()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        src_key_padding_mask = (attention_mask == 0)

        encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        return type('Output', (), {'last_hidden_state': encoded})()

class TextEncoder(nn.Module):
    """Text encoder using pre-trained language models to encode metadata, sleep data and food data"""

    def __init__(self, model_key=None, hidden_dim=256, use_mirror=True, qwen_model_path=None):
        """
        Initialize text encoder

        Args:
            model_key: Pre-trained language model key
            hidden_dim: Hidden layer dimension
            use_mirror: Whether to use HF mirror
            qwen_model_path: Local path for Qwen model
        """
        super(TextEncoder, self).__init__()

        self.use_mirror = use_mirror
        self.hidden_dim = hidden_dim

        try:
            from model_config import get_model_config
            config = get_model_config(model_key)
            model_name = config['name']
        except:
            model_name = 'bert-base-chinese'

        if use_mirror:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        try:
            if qwen_model_path is not None:
                self.language_model = AutoModel.from_pretrained(qwen_model_path, trust_remote_code=True)
                print(f"Loading text model from provided local path: {qwen_model_path}")
            elif 'Embedding' in model_name:
                if 'Qwen3-Embedding' in model_name:
                    try:
                        from huggingface_hub import snapshot_download
                        local_path = snapshot_download(
                            repo_id=model_name,
                            endpoint='https://hf-mirror.com'
                        )
                        self.language_model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
                        print(f"Loading embedding model from local path: {local_path}")
                    except:
                        self.language_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                        print(f"Downloading embedding model using HF mirror: {model_name}")
                else:
                    self.language_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    print(f"Downloading embedding model using HF mirror: {model_name}")
            else:
                self.language_model = AutoModel.from_pretrained(model_name)
                print(f"Downloading model using HF mirror: {model_name}")
        except Exception as e:
            print(f"Failed to load pre-trained model {model_name}: {str(e)}")
            self.language_model = SimpleLanguageModel(
                vocab_size=21128,
                hidden_size=768,
                num_layers=6,
                num_heads=12
            )

        lm_hidden_dim = self.language_model.config.hidden_size

        self.metadata_projection = nn.Sequential(
            nn.Linear(lm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.sleep_projection = nn.Sequential(
            nn.Linear(lm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.food_projection = nn.Sequential(
            nn.Linear(lm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.exercise_projection = nn.Sequential(
            nn.Linear(lm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, metadata_input_ids, metadata_attention_mask,
                sleep_input_ids, sleep_attention_mask,
                food_input_ids, food_attention_mask,
                exercise_input_ids=None, exercise_attention_mask=None):
        """
        Forward pass

        Args:
            metadata_input_ids: Metadata input token IDs [batch_size, seq_len]
            metadata_attention_mask: Metadata attention mask [batch_size, seq_len]
            sleep_input_ids: Sleep data input token IDs [batch_size, seq_len]
            sleep_attention_mask: Sleep data attention mask [batch_size, seq_len]
            food_input_ids: Food data input token IDs [batch_size, seq_len]
            food_attention_mask: Food data attention mask [batch_size, seq_len]
            exercise_input_ids: Exercise data input token IDs [batch_size, seq_len]
            exercise_attention_mask: Exercise data attention mask [batch_size, seq_len]

        Returns:
            encoded_metadata: Encoded metadata features [batch_size, hidden_dim]
            encoded_sleep: Encoded sleep data features [batch_size, hidden_dim]
            encoded_food: Encoded food data features [batch_size, hidden_dim]
            encoded_exercise: Encoded exercise data features [batch_size, hidden_dim]
        """
        def pool_valid_tokens(outputs, attention_mask):
            """Pool all non-padding token states.

            The Qwen checkpoints use a causal mask, so position zero is not a
            CLS token and cannot summarize text that follows it.  A masked mean
            keeps every valid token while excluding left/right padding and also
            works with the small fallback language model used in tests.
            """
            hidden = outputs.last_hidden_state
            if attention_mask is None:
                return hidden.mean(dim=1)
            weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        metadata_outputs = self.language_model(
            input_ids=metadata_input_ids,
            attention_mask=metadata_attention_mask
        )
        metadata_summary = pool_valid_tokens(metadata_outputs, metadata_attention_mask)
        encoded_metadata = self.metadata_projection(metadata_summary)

        sleep_outputs = self.language_model(
            input_ids=sleep_input_ids,
            attention_mask=sleep_attention_mask
        )
        sleep_summary = pool_valid_tokens(sleep_outputs, sleep_attention_mask)
        encoded_sleep = self.sleep_projection(sleep_summary)

        food_outputs = self.language_model(
            input_ids=food_input_ids,
            attention_mask=food_attention_mask
        )
        food_summary = pool_valid_tokens(food_outputs, food_attention_mask)
        encoded_food = self.food_projection(food_summary)

        # Encode exercise data
        if exercise_input_ids is not None and exercise_attention_mask is not None:
            exercise_outputs = self.language_model(
                input_ids=exercise_input_ids,
                attention_mask=exercise_attention_mask
            )
            exercise_summary = pool_valid_tokens(exercise_outputs, exercise_attention_mask)
            encoded_exercise = self.exercise_projection(exercise_summary)
        else:
            batch_size = metadata_input_ids.size(0)
            encoded_exercise = torch.zeros(batch_size, self.hidden_dim).to(metadata_input_ids.device)

        return encoded_metadata, encoded_sleep, encoded_food, encoded_exercise

class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CGMTransformerEncoder(nn.Module):
    """Transformer encoder for CGM data"""

    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        """
        Initialize CGM Transformer encoder

        Args:
            input_dim: Input feature dimension (CGM value is 1)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout rate
        """
        super(CGMTransformerEncoder, self).__init__()

        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, cgm_sequence, attention_mask=None):
        """
        Forward pass

        Args:
            cgm_sequence: CGM sequence [batch_size, seq_len]
            attention_mask: 1 for measured points and 0 for padding [batch_size, seq_len]

        Returns:
            encoded_cgm: Encoded CGM features [batch_size, seq_len, d_model]
        """
        cgm_sequence = cgm_sequence.unsqueeze(-1)

        x = self.input_projection(cgm_sequence)

        x = x.transpose(0, 1)

        x = self.pos_encoder(x)

        x = x.transpose(0, 1)

        key_padding_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            key_padding_mask = ~valid_mask
            # PyTorch's attention can produce NaNs when every position in a
            # row is masked.  Keep an all-zero placeholder row unmasked; the
            # caller's masked pooling/loss still excludes it from learning.
            empty_rows = ~valid_mask.any(dim=1)
            if empty_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[empty_rows] = False
        encoded_cgm = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        return encoded_cgm

class CGMTransformerDecoder(nn.Module):
    """Transformer decoder for CGM data, used to predict postprandial CGM"""

    def __init__(self, d_model=128, output_dim=1, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        """
        Initialize CGM Transformer decoder

        Args:
            d_model: Model dimension
            output_dim: Output dimension (CGM value is 1)
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout rate
        """
        super(CGMTransformerDecoder, self).__init__()

        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model)

        # CGM values are scalars; project them before adding d_model-wide
        # positional encodings.  This also makes teacher forcing and
        # autoregressive inference use exactly the same representation.
        self.target_input_projection = nn.Linear(output_dim, d_model)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, encoded_cgm, encoded_text, target_sequence=None):
        """
        Forward pass

        Args:
            encoded_cgm: Encoded CGM features [batch_size, seq_len, d_model]
            encoded_text: Encoded text features [batch_size, d_model]
            target_sequence: Target sequence (used during training) [batch_size, target_len, 1]

        Returns:
            predicted_cgm: Predicted CGM sequence [batch_size, target_len, 1]
        """
        batch_size = encoded_cgm.size(0)
        seq_len = encoded_cgm.size(1)

        if target_sequence is not None:
            target_len = target_sequence.size(1)

            # Teacher forcing uses only the start token and targets strictly
            # before the current position.  The target at t is therefore
            # never available when predicting y_t.
            decoder_inputs = torch.zeros_like(target_sequence)
            if target_len > 1:
                decoder_inputs[:, 1:, :] = target_sequence[:, :-1, :]
            target_embedding = self.target_input_projection(decoder_inputs)
            target_embedding = self.pos_encoder(target_embedding.transpose(0, 1)).transpose(0, 1)

            memory = encoded_text.unsqueeze(1).expand(-1, target_len, -1)
            causal_mask = torch.triu(
                torch.ones(target_len, target_len, device=target_embedding.device, dtype=torch.bool),
                diagonal=1,
            )
            decoded = self.transformer_decoder(
                tgt=target_embedding,
                memory=memory,
                tgt_mask=causal_mask,
            )

            predicted_cgm = self.output_projection(decoded)

        else:
            target_len = seq_len

            output_sequence = torch.zeros(batch_size, target_len, 1, device=encoded_cgm.device)

            for i in range(target_len):
                current_seq = output_sequence[:, :i+1, :]

                current_seq = self.target_input_projection(current_seq)
                current_seq = self.pos_encoder(current_seq.transpose(0, 1)).transpose(0, 1)

                memory = encoded_text.unsqueeze(1).expand(-1, i+1, -1)

                decoded = self.transformer_decoder(
                    tgt=current_seq,
                    memory=memory
                )

                next_value = self.output_projection(decoded[:, -1:, :])
                output_sequence[:, i:i+1, :] = next_value

            predicted_cgm = output_sequence

        return predicted_cgm

class ImageEncoder(nn.Module):
    """Image encoder for the configured diet vision-language model.

    The public demo can still use ``cnn-backup``.  For research runs the
    configuration resolves to the full Qwen2.5-VL-72B-Instruct repository id,
    and an already downloaded local directory is preferred when supplied.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-VL-72B-Instruct", hidden_dim=256,
                 use_mirror=True, force_cnn=False, model_path=None):
        """
        Initialize image encoder

        Args:
            model_name: Pre-trained vision-language model name
            hidden_dim: Hidden layer dimension
            use_mirror: Whether to use HF mirror
            force_cnn: Whether to force use CNN (don't try to load vision model)
        """
        super(ImageEncoder, self).__init__()

        self.use_mirror = use_mirror
        self.force_cnn = force_cnn
        self.model_path = model_path

        # Accept either a short config key or a full repository id.  Passing a
        # short key directly to from_pretrained() was the source of the old
        # Qwen3-VL/embedding mismatch.
        try:
            from model_config import VISION_MODELS, get_vision_model_config
            if model_name in VISION_MODELS:
                resolved = get_vision_model_config(model_name)
                model_name = resolved['name']
                if model_name == 'CNN Backup':
                    force_cnn = True
        except Exception:
            pass
        self.force_cnn = force_cnn
        self.model_name = model_name

        if use_mirror:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        if force_cnn:
            print("Force using CNN encoder")
            self.vision_model = None
            self.processor = None
            vision_hidden_dim = 512
            self._init_cnn_encoder(vision_hidden_dim, hidden_dim)
        else:
            try:
                load_name = model_path or model_name
                print(f"Loading vision-language model: {load_name}")
                import socket
                socket.setdefaulttimeout(5)

                self.vision_model = AutoModel.from_pretrained(
                    load_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    local_files_only=False
                )
                self.processor = AutoProcessor.from_pretrained(
                    load_name,
                    trust_remote_code=True,
                    local_files_only=False
                )
                print(f"Successfully loaded vision-language model: {load_name}")

                if hasattr(self.vision_model.config, 'hidden_size'):
                    vision_hidden_dim = self.vision_model.config.hidden_size
                elif hasattr(self.vision_model, 'visual') and hasattr(self.vision_model.visual.config, 'hidden_size'):
                    vision_hidden_dim = self.vision_model.visual.config.hidden_size
                else:
                    vision_hidden_dim = 1024

            except (Exception, socket.timeout) as e:
                print(f"Failed to load vision-language model {model_name}: {str(e)}")
                print("Network unavailable or model loading failed, using CNN encoder")
                self.vision_model = None
                self.processor = None
                vision_hidden_dim = 512
                self._init_cnn_encoder(vision_hidden_dim, hidden_dim)

        self.image_projection = nn.Sequential(
            nn.Linear(vision_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _init_cnn_encoder(self, vision_hidden_dim, hidden_dim):
        """Initialize CNN encoder"""
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, vision_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        print("Using simple CNN encoder")

    def forward(self, images):
        """
        Forward pass

        Args:
            images: List of image tensors or PIL images

        Returns:
            encoded_images: Encoded image features [batch_size, hidden_dim]
        """
        batch_size = len(images)

        if self.vision_model is not None and self.processor is not None:
            try:
                if isinstance(images[0], torch.Tensor):
                    pil_images = []
                    # ``SampleImagePredictionDataset`` normalizes tensors for
                    # the CNN path.  Restore RGB values before handing them to
                    # the VLM processor, which performs its own preprocessing.
                    mean = torch.tensor([0.485, 0.456, 0.406], device=images[0].device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=images[0].device).view(3, 1, 1)
                    for img_tensor in images:
                        rgb_tensor = (img_tensor * std + mean).clamp(0, 1)
                        img_np = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(img_np))
                else:
                    pil_images = images

                inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt"
                )

                device = next(self.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                vision_trainable = any(p.requires_grad for p in self.vision_model.parameters())
                if vision_trainable:
                    outputs = self.vision_model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = self.vision_model(**inputs)

                if hasattr(outputs, 'image_embeds'):
                    image_features = outputs.image_embeds
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use the full image-token mean when the model exposes a
                    # sequence, avoiding a BERT-style CLS assumption.
                    image_features = outputs.last_hidden_state.mean(dim=1)
                else:
                    image_features = outputs.last_hidden_state.mean(dim=1)

                if image_features.dim() == 3:
                    image_features = image_features.mean(dim=1)

            except Exception as e:
                print(f"Vision model processing failed: {str(e)}")
                image_features = self._fallback_process(images)
        else:
            image_features = self._fallback_process(images)

        encoded_images = self.image_projection(image_features)

        return encoded_images

    def _fallback_process(self, images):
        """Fallback processing method using simple CNN"""
        batch_size = len(images)
        device = next(self.parameters()).device

        if isinstance(images[0], torch.Tensor):
            image_tensors = torch.stack(images).to(device)
            if image_tensors.max() > 1.0:
                image_tensors = image_tensors / 255.0
        else:
            image_tensors = []
            for pil_img in images:
                tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
                if tensor.dim() == 3:
                    tensor = tensor.permute(2, 0, 1)
                image_tensors.append(tensor)
            image_tensors = torch.stack(image_tensors).to(device)

        if image_tensors.shape[1] == 1:
            image_tensors = image_tensors.repeat(1, 3, 1, 1)
        elif image_tensors.shape[1] == 4:
            image_tensors = image_tensors[:, :3, :, :]

        image_features = self.cnn_encoder(image_tensors)

        return image_features


class ActionTransitionModule(nn.Module):
    """Action-conditioned latent transition with trajectory and dynamics branches."""

    def __init__(self, state_dim, action_dim, dropout=0.1):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, state_dim)
        self.transition = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trajectory_branch = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim)
        )
        self.dynamics_branch = nn.GRUCell(state_dim, state_dim)

    def forward(self, physiological_state, action_state):
        action_state = self.action_projection(action_state)
        intermediate = self.transition(torch.cat([physiological_state, action_state], dim=-1))
        dynamic_state = self.dynamics_branch(intermediate, physiological_state)
        trajectory_state = self.trajectory_branch(dynamic_state)
        return dynamic_state, trajectory_state


class CrossAttentionFusion(nn.Module):
    """Fuse diet and exercise future states with cross-attention."""

    def __init__(self, state_dim, nhead=4, dropout=0.1):
        super().__init__()
        if state_dim % nhead:
            nhead = 1
        self.attention = nn.MultiheadAttention(state_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, diet_state, exercise_state):
        diet_tokens = diet_state.unsqueeze(1)
        exercise_tokens = exercise_state.unsqueeze(1)
        attended, _ = self.attention(diet_tokens, exercise_tokens, exercise_tokens)
        return self.norm((diet_tokens + attended).squeeze(1))


class PrescriptionRankingHead(nn.Module):
    """Decode a future glucose proxy and a quantitative candidate score."""

    def __init__(self, state_dim, max_future_len=32):
        super().__init__()
        self.glucose_head = nn.Linear(state_dim, max_future_len)
        self.score_head = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, 1)
        )

    def forward(self, fused_state, target_len=None):
        glucose = self.glucose_head(fused_state)
        if target_len is not None:
            if target_len <= glucose.size(1):
                glucose = glucose[:, :target_len]
            else:
                tail = glucose[:, -1:].expand(-1, target_len - glucose.size(1))
                glucose = torch.cat([glucose, tail], dim=1)
        score = self.score_head(fused_state).squeeze(-1)
        return glucose, score


class CGMPredictionModelWithImage(nn.Module):
    """Action-conditioned CGM world model with optional ranking outputs."""

    def __init__(self,
                 model_key=None,
                 text_hidden_dim=256,
                 cgm_d_model=128,
                 cgm_nhead=8,
                 cgm_num_layers=3,
                 cgm_dim_feedforward=512,
                 dropout=0.1,
                 use_mirror=True,
                 qwen_model_path=None,
                 use_image_features=True,
                 vision_model_key="qwen2.5-vl-72b-instruct",
                 vision_model_path=None,
                 max_future_len=32):
        """
        Initialize CGM prediction model

        Args:
            model_key: Language model key
            text_hidden_dim: Text feature dimension
            cgm_d_model: CGM Transformer model dimension
            cgm_nhead: Number of attention heads
            cgm_num_layers: Number of Transformer layers
            cgm_dim_feedforward: Feed-forward network dimension
            dropout: Dropout rate
            use_mirror: Whether to use HF mirror
            qwen_model_path: Local path for Qwen model
            use_image_features: Whether to use image features
        """
        super(CGMPredictionModelWithImage, self).__init__()

        self.use_image_features = use_image_features
        self.vision_model_path = vision_model_path
        self.max_future_len = max_future_len

        self.text_encoder = TextEncoder(
            model_key=model_key,
            hidden_dim=text_hidden_dim,
            use_mirror=use_mirror,
            qwen_model_path=qwen_model_path
        )

        if self.use_image_features:
            force_cnn = (vision_model_key == 'cnn-backup')

            self.image_encoder = ImageEncoder(
                model_name=vision_model_key,
                hidden_dim=text_hidden_dim,
                use_mirror=use_mirror,
                force_cnn=force_cnn,
                model_path=vision_model_path
            )

        self.cgm_encoder = CGMTransformerEncoder(
            input_dim=1,
            d_model=cgm_d_model,
            nhead=cgm_nhead,
            num_layers=cgm_num_layers,
            dim_feedforward=cgm_dim_feedforward,
            dropout=dropout
        )

        fusion_input_dim = text_hidden_dim * 4 + cgm_d_model
        if self.use_image_features:
            fusion_input_dim += text_hidden_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cgm_d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Manuscript-aligned diet/exercise sub-world models and ranking path.
        diet_action_dim = text_hidden_dim * (2 if self.use_image_features else 1)
        self.diet_action_projection = nn.Linear(diet_action_dim, cgm_d_model)
        self.exercise_action_projection = nn.Linear(text_hidden_dim, cgm_d_model)
        self.diet_transition = ActionTransitionModule(cgm_d_model, cgm_d_model, dropout)
        self.exercise_transition = ActionTransitionModule(cgm_d_model, cgm_d_model, dropout)
        self.future_state_fusion = CrossAttentionFusion(cgm_d_model, nhead=cgm_nhead, dropout=dropout)
        self.ranking_head = PrescriptionRankingHead(cgm_d_model, max_future_len=max_future_len)

        self.cgm_decoder = CGMTransformerDecoder(
            d_model=cgm_d_model,
            output_dim=1,
            nhead=cgm_nhead,
            num_layers=cgm_num_layers,
            dim_feedforward=cgm_dim_feedforward,
            dropout=dropout
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to(self, device):
        """Override to method to ensure all submodules move to correct device"""
        super(CGMPredictionModelWithImage, self).to(device)
        self.device = device
        return self

    def forward(self, metadata_input_ids, metadata_attention_mask,
                sleep_input_ids, sleep_attention_mask,
                food_input_ids, food_attention_mask,
                exercise_input_ids=None, exercise_attention_mask=None,
                cgm_preprandial=None, cgm_postprandial=None,
                food_images=None, cgm_preprandial_mask=None,
                return_details=False):
        """
        Forward pass

        Args:
            metadata_input_ids: Metadata input token IDs [batch_size, seq_len]
            metadata_attention_mask: Metadata attention mask [batch_size, seq_len]
            sleep_input_ids: Sleep data input token IDs [batch_size, seq_len]
            sleep_attention_mask: Sleep data attention mask [batch_size, seq_len]
            food_input_ids: Food data input token IDs [batch_size, seq_len]
            food_attention_mask: Food data attention mask [batch_size, seq_len]
            exercise_input_ids: Exercise data input token IDs [batch_size, seq_len]
            exercise_attention_mask: Exercise data attention mask [batch_size, seq_len]
            cgm_preprandial: Preprandial CGM sequence [batch_size, seq_len]
            cgm_preprandial_mask: 1 for measured preprandial points and 0 for
                padding [batch_size, seq_len]
            cgm_postprandial: Postprandial CGM sequence (used during training) [batch_size, target_len]
            food_images: Food image list [batch_size]

        Returns:
            predicted_cgm: Predicted postprandial CGM sequence [batch_size, target_len, 1]
        """
        encoded_metadata, encoded_sleep, encoded_food, encoded_exercise = self.text_encoder(
            metadata_input_ids, metadata_attention_mask,
            sleep_input_ids, sleep_attention_mask,
            food_input_ids, food_attention_mask,
            exercise_input_ids, exercise_attention_mask
        )

        if self.use_image_features and food_images is not None:
            encoded_images = self.image_encoder(food_images)
        else:
            batch_size = metadata_input_ids.size(0)
            encoded_images = torch.zeros(batch_size, encoded_metadata.size(1)).to(self.device)

        encoded_cgm = self.cgm_encoder(cgm_preprandial, attention_mask=cgm_preprandial_mask)

        if cgm_preprandial_mask is None:
            pooled_cgm = torch.mean(encoded_cgm, dim=1)
        else:
            cgm_weights = cgm_preprandial_mask.to(
                device=encoded_cgm.device, dtype=encoded_cgm.dtype
            ).unsqueeze(-1)
            pooled_cgm = (encoded_cgm * cgm_weights).sum(dim=1) / cgm_weights.sum(dim=1).clamp_min(1.0)

        if self.use_image_features:
            fused_features = torch.cat([
                encoded_metadata, encoded_sleep, encoded_food, encoded_exercise,
                encoded_images, pooled_cgm
            ], dim=1)
        else:
            fused_features = torch.cat([
                encoded_metadata, encoded_sleep, encoded_food, encoded_exercise,
                pooled_cgm
            ], dim=1)

        fused_features = self.feature_fusion(fused_features)

        # Generate separate diet and exercise future states before the
        # cross-attention fusion and ranking heads described in the manuscript.
        if self.use_image_features:
            diet_action = self.diet_action_projection(torch.cat([encoded_food, encoded_images], dim=-1))
        else:
            diet_action = self.diet_action_projection(encoded_food)
        exercise_action = self.exercise_action_projection(encoded_exercise)
        diet_dynamic, diet_trajectory = self.diet_transition(fused_features, diet_action)
        exercise_dynamic, exercise_trajectory = self.exercise_transition(fused_features, exercise_action)
        future_state = self.future_state_fusion(diet_trajectory, exercise_trajectory)

        if cgm_postprandial is not None:
            target_sequence = cgm_postprandial.unsqueeze(-1)
            predicted_cgm = self.cgm_decoder(encoded_cgm, future_state, target_sequence)
            target_len = target_sequence.size(1)
        else:
            predicted_cgm = self.cgm_decoder(encoded_cgm, future_state)
            target_len = predicted_cgm.size(1)

        ranking_glucose, ranking_score = self.ranking_head(future_state, target_len=target_len)
        if return_details:
            return {
                'predicted_cgm': predicted_cgm.squeeze(-1),
                'ranking_score': ranking_score,
                'ranking_glucose': ranking_glucose,
                'diet_dynamic_state': diet_dynamic,
                'diet_trajectory_state': diet_trajectory,
                'exercise_dynamic_state': exercise_dynamic,
                'exercise_trajectory_state': exercise_trajectory,
                'fused_future_state': future_state,
            }

        return predicted_cgm.squeeze(-1)  # [batch_size, target_len]
