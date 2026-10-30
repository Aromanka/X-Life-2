import math
import os
import torch
from torch import nn
from model import CGMTransformerEncoder, transformer_encoder, causal_mask
from backends.metadata_encoder import SemanticMetadataEncoder
from configuration import configured, settings
from action_inputs import action_inputs


def ranking_probabilities(logits):
    """Normalize one complete candidate set sharing a single user state."""
    if logits.ndim != 1 or not logits.numel(): raise ValueError('A nonempty candidate logit vector is required')
    if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite ranking logit')
    return torch.softmax(logits, dim=0)


def llm_actions_enabled():
    value = os.environ.get('XLIFE_USE_LLM_ACTION_ENCODERS')
    if value is None:
        return settings('http', 'use_llm_action_encoders')
    value = value.strip().lower()
    if value not in {'', '0', 'false', 'no', 'off', '1', 'true', 'yes', 'on'}: raise ValueError('Invalid XLIFE_USE_LLM_ACTION_ENCODERS')
    return value in {'1', 'true', 'yes', 'on'}


class NumericEncoder(nn.Module):
    def __init__(self, size, dimension):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('scale', torch.ones(size))
        self.network = nn.Sequential(nn.Linear(size * 2, dimension), nn.GELU(), nn.Linear(dimension, dimension))

    def fit(self, values):
        valid = torch.isfinite(values)
        count = valid.sum(0).clamp_min(1)
        mean = torch.where(valid, values, 0).sum(0) / count
        scale = (torch.where(valid, (values-mean).square(), 0).sum(0)/count).sqrt().clamp_min(1e-3)
        self.mean.copy_(mean); self.scale.copy_(scale)

    def forward(self, values):
        valid = torch.isfinite(values)
        normalized = torch.where(valid, (values-self.mean)/self.scale, 0)
        return self.network(torch.cat([normalized, valid.to(values.dtype)], -1))


class Transition(nn.Module):
    def __init__(self, dimension, basis_dim, heads, layers, architecture):
        super().__init__()
        self.conditioning = nn.Sequential(nn.Linear(2*dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.basis = nn.Parameter(torch.empty(basis_dim, dimension))
        nn.init.orthogonal_(self.basis)
        self.dynamics = nn.Sequential(nn.Linear(basis_dim, dimension), nn.LayerNorm(dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.trajectory = transformer_encoder(dimension, heads, layers, architecture['dropout'],
            dimension*architecture['feedforward_multiplier'], architecture['activation'])
        self.decoder = nn.Linear(dimension, 1)

    def increment(self, state): return self.dynamics(state @ self.basis.T)

    def readout(self, states):
        """Read Q_n from S_0..S_n with no access to S_(n+1)."""
        q = self.trajectory(states, mask=causal_mask(states.size(1), states.device))
        return q, self.decoder(q).squeeze(-1)


class MetabolicWorldModel(nn.Module):
    """Residual diet, exercise and gated joint dynamics."""
    def __init__(self, dimension=None, basis_dim=None, heads=None, layers=None, patch_size=None, diet_model_path=None, exercise_model_path=None,
                 action_backend_id=None, state_pooling=None, gap_feature=None, readout=None, schema=None,
                 dropout=None, feedforward=None, feedforward_multiplier=None, type_capacity=None,
                 position_max_length=None, time_encoding_base=None, activation=None, data_config=None):
        super().__init__()
        dimension = configured('model', 'dimension', dimension)
        basis_dim = configured('model', 'basis_dim', basis_dim)
        heads = configured('model', 'heads', heads)
        layers = configured('model', 'layers', layers)
        patch_size = configured('model', 'patch_size', patch_size)
        action_backend_id = configured('model', 'action_backend_id', action_backend_id)
        state_pooling = configured('model', 'state_pooling', state_pooling)
        gap_feature = configured('model', 'gap_feature', gap_feature)
        readout = configured('model', 'readout', readout)
        self.model_settings = {key: configured('model', key, value) for key, value in
                              dict(dropout=dropout, feedforward=feedforward, feedforward_multiplier=feedforward_multiplier,
                                   type_capacity=type_capacity, position_max_length=position_max_length,
                                   time_encoding_base=time_encoding_base, activation=activation).items()}
        self.data_config = settings('data') if data_config is None else data_config
        if dimension < 1 or basis_dim < 1 or heads < 1 or layers < 1 or patch_size < 1:
            raise ValueError('Model dimensions and counts must be positive')
        if diet_model_path or exercise_model_path:
            raise ValueError('Local language-model paths are retired; configure the external HTTP role service')
        self.dimension, self.llm_actions = dimension, llm_actions_enabled()
        self.schema = schema
        self.supervision_heads = nn.ModuleDict()
        for spec in (schema or {}).get('supervision', []):
            if spec.get('prediction'): continue
            shape = spec['shape'][1:] if spec['readout'] == 'sequence' else spec['shape']
            self.supervision_heads[spec['name']] = nn.ModuleDict({mode:nn.Linear(dimension * (2 if mode == 'joint' else 1), math.prod(shape)) for mode in ('diet','exercise','joint')})
        self.diet_encoder, self.exercise_encoder = NumericEncoder(6, dimension), NumericEncoder(3, dimension)
        self.metadata_encoder = SemanticMetadataEncoder(dimension)
        self.type_embedding = nn.Embedding(self.model_settings['type_capacity'], dimension)
        self.cgm_encoder = CGMTransformerEncoder(d_model=dimension, nhead=heads, num_layers=layers,
            dropout=self.model_settings['dropout'], patch_size=patch_size, dim_feedforward=self.model_settings['feedforward'],
            position_max_length=self.model_settings['position_max_length'], activation=self.model_settings['activation'],
            time_encoding_base=self.model_settings['time_encoding_base'])
        self.state_projection = nn.Sequential(nn.Linear(3*dimension+2, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.diet = Transition(dimension, basis_dim, heads, layers, self.model_settings)
        self.exercise = Transition(dimension, basis_dim, heads, layers, self.model_settings)
        self.fusion = Transition(2*dimension, 2*basis_dim, heads, layers, self.model_settings)
        self.gate = nn.Linear(2*basis_dim, 3)
        self.register_buffer('glucose_mean', torch.tensor(0.))
        self.register_buffer('glucose_scale', torch.tensor(1.))
        if state_pooling not in ('both','pooled','recent') or readout not in ('direct','autoregressive'):
            raise ValueError('Unknown controlled ablation')
        self.state_pooling, self.gap_feature, self.readout = state_pooling, gap_feature, readout
        if readout == 'autoregressive':
            from model import CGMTransformerDecoder
            self.ar_decoders = nn.ModuleDict({name:CGMTransformerDecoder(d_model=dimension*(2 if name=='joint' else 1), nhead=heads, num_layers=layers,
                dropout=self.model_settings['dropout'], dim_feedforward=self.model_settings['feedforward'],
                position_max_length=self.model_settings['position_max_length'], activation=self.model_settings['activation'],
                time_encoding_base=self.model_settings['time_encoding_base']) for name in ('diet','exercise','joint')})
        if self.llm_actions:
            if not action_backend_id: raise ValueError('Configure action_backend_id for HTTP action encoders')
            if isinstance(action_backend_id, str):
                action_ids = {domain: action_backend_id for domain in ('diet', 'exercise')}
            elif isinstance(action_backend_id, dict) and set(action_backend_id) == {'diet', 'exercise'}:
                action_ids = action_backend_id
            else: raise ValueError('action_backend_id requires a shared string or diet/exercise artifact IDs')
            if any(not isinstance(value, str) or not value.strip() for value in action_ids.values()):
                raise ValueError('Action artifact IDs must be nonempty strings')
            from backends.action_encoder import HTTPActionEncoder
            self.diet_vlm = HTTPActionEncoder('diet', dimension, action_ids['diet'])
            self.exercise_vlm = HTTPActionEncoder('exercise', dimension, action_ids['exercise'])
        self.ranking_head = RankingHead(dimension, heads, layers, self.model_settings)
        self.register_buffer('ranked_modes', torch.zeros(3, dtype=torch.bool))

    def fit(self, samples):
        for name in ('diet', 'exercise'):
            getattr(self, name+'_encoder').fit(torch.stack([s[name] for s in samples]))
        with torch.no_grad(): self.metadata_encoder([samples[0]['metadata']])
        observed = torch.cat([s['history'][s['history_mask']] for s in samples])
        self.glucose_mean.copy_(observed.mean()); self.glucose_scale.copy_(observed.std(unbiased=False).clamp_min(1e-3))

    def time_encoding(self, minutes, dimension):
        frequencies = torch.exp(torch.arange(0, dimension, 2, device=minutes.device)*(-math.log(self.model_settings['time_encoding_base'])/dimension))
        phase = minutes[:, None]/self.data_config['interval_minutes'] * frequencies
        return torch.stack([phase.sin(), phase.cos()], -1).flatten(-2)[:, :dimension]

    def initial_states(self, batch, mode):
        mask, history = batch['history_mask'], batch['history']
        values = (history-self.glucose_mean)/self.glucose_scale
        encoded = self.cgm_encoder(values, mask)
        weights = self.cgm_encoder.patch_weights(values, mask).unsqueeze(-1)
        pooled = (encoded*weights).sum(1)/weights.sum(1).clamp_min(1)
        positions = torch.arange(history.size(1), device=history.device).expand_as(history)
        last_index = positions.masked_fill(~mask, -1).max(1).values
        if (last_index < 0).any(): raise ValueError('History requires at least one measured value')
        last = history.gather(1, last_index[:, None]).squeeze(1)
        recent = encoded[torch.arange(len(history), device=history.device), last_index//self.cgm_encoder.patch_size]
        if self.state_pooling == 'pooled': recent = torch.zeros_like(recent)
        if self.state_pooling == 'recent': pooled = torch.zeros_like(pooled)
        gap = (history.size(1)-1-last_index)[:, None]/history.size(1) if self.gap_feature else torch.zeros_like(last_index[:,None])
        state = self.state_projection(torch.cat([pooled, recent, self.metadata_encoder(batch['metadata']), ((last-self.glucose_mean)/self.glucose_scale)[:, None], gap], -1))
        inputs = []
        if self.llm_actions:
            for i in range(len(history)):
                action = {key:batch[key][i].detach().cpu().tolist() for key in ('diet', 'exercise', 'onset')}
                action.update(mode=mode, diet_text=batch['diet_text'][i], exercise_text=batch['exercise_text'][i], diet_images=batch.get('diet_images', [[] for _ in history])[i])
                inputs.append(action_inputs(action))
        states = {}
        for index, domain in enumerate(('diet', 'exercise')):
            if mode not in (domain, 'joint'): continue
            if self.llm_actions:
                texts = [row[domain+'_text'] for row in inputs]
                action = self.diet_vlm(texts, [row['diet_images'] for row in inputs]) if domain == 'diet' else self.exercise_vlm(texts)
            else:
                action = getattr(self, domain+'_encoder')(batch[domain])
                if domain == 'exercise': action = action + self.type_embedding(batch['type_id'])
            action = action + self.time_encoding(batch['onset'][:, index], self.dimension)
            states[domain] = getattr(self, domain).conditioning(torch.cat([state, action], -1))
        return states.get('diet'), states.get('exercise'), last

    def initialize_fusion(self):
        with torch.no_grad(): self.fusion.basis.copy_(torch.block_diag(self.diet.basis, self.exercise.basis))

    def configure_phase(self, phase):
        if phase not in {'single', 'fusion', 'professional', 'physiological', 'user_feedback'}: raise ValueError('Invalid training phase')
        for parameter in self.parameters(): parameter.requires_grad = False
        if phase in {'single', 'fusion'}: self.ranked_modes.zero_()
        names = ('diet_encoder', 'exercise_encoder', 'type_embedding', 'metadata_encoder', 'cgm_encoder', 'state_projection', 'diet', 'exercise') if phase == 'single' else ('fusion', 'gate') if phase == 'fusion' else ('ranking_head',)
        for name in names:
            for parameter in getattr(self, name).parameters(): parameter.requires_grad = True
        if phase == 'user_feedback':
            for mode in ('diet', 'exercise', 'joint'):
                self.glucose_simulation_head(mode).requires_grad_(True)
        if phase in ('single', 'fusion'):
            for heads in self.supervision_heads.values():
                for mode in (('joint',) if phase == 'fusion' else ('diet','exercise')): heads[mode].requires_grad_(True)
        if phase == 'fusion':
            self.diet.conditioning.requires_grad_(True); self.exercise.conditioning.requires_grad_(True)
        if self.llm_actions and phase == 'single':
            for encoder in (self.diet_vlm, self.exercise_vlm):
                for parameter in encoder.projection.parameters(): parameter.requires_grad = True
        if self.readout == 'autoregressive' and phase in ('single','fusion'):
            for domain in (('joint',) if phase == 'fusion' else ('diet','exercise')): self.ar_decoders[domain].requires_grad_(True)

    def glucose_simulation_head(self, mode):
        """The glucose readout, excluding shared trajectory and dynamics parameters."""
        if mode not in ('diet', 'exercise', 'joint'): raise ValueError('Invalid simulation mode')
        if self.readout == 'autoregressive': return self.ar_decoders[mode]
        return (self.fusion if mode == 'joint' else getattr(self, mode)).decoder

    def train(self, mode=True):
        super().train(mode)
        if self.llm_actions: self.diet_vlm.eval(); self.exercise_vlm.eval()
        return self

    def forward(self, batch, mode, horizon=None):
        horizon = self.data_config['horizon_steps'] if horizon is None else horizon
        if mode not in {'diet', 'exercise', 'joint'} or horizon < 1: raise ValueError('Invalid mode or horizon')
        diet, exercise, last = self.initial_states(batch, mode)
        state = torch.cat([diet, exercise], -1) if mode == 'joint' else diet if mode == 'diet' else exercise
        states, gates = [], []
        for _ in range(horizon):
            # Both branches consume S_n: readout produces Q_n, while the
            # residual dynamics below advance the rollout to S_(n+1).
            states.append(state)
            if mode == 'joint':
                d, e = state.split(self.dimension, -1)
                weights = torch.sigmoid(self.gate(state @ self.fusion.basis.T))
                dm, em = self.diet.increment(d), self.exercise.increment(e)
                state = state + torch.cat([weights[:, :1]*dm, weights[:, 1:2]*em], -1) + weights[:, 2:3]*self.fusion.increment(state)
                gates.append(weights)
            else: state = state + getattr(self, mode).increment(state)
        states = torch.stack(states, 1)
        q, glucose = (self.fusion if mode == 'joint' else getattr(self, mode)).readout(states)
        if self.readout == 'autoregressive': glucose = self.ar_decoders[mode](None, q).squeeze(-1)
        return dict(predicted_cgm=last[:, None]+glucose*self.glucose_scale, trajectory=q, ranking_logit=self.ranking_head(q, mode), states=states, gates=torch.stack(gates, 1) if gates else None)

    @torch.no_grad()
    def rank(self, batch, modes):
        """Return 0–1 scores for a complete candidate set with one shared context."""
        return ranking_probabilities(self.rank_logits(batch, modes))

    @torch.no_grad()
    def rank_logits(self, batch, modes):
        """Return unbounded logits; callers may collect chunks before softmax."""
        if len(modes) != len(batch['history']): raise ValueError('One mode per candidate is required')
        scores = batch['history'].new_empty(len(modes))
        for mode in set(modes):
            if mode not in ('diet', 'exercise', 'joint') or not self.ranked_modes[('diet', 'exercise', 'joint').index(mode)]: raise ValueError(f'No trained ranking for {mode}')
            indices = [i for i, value in enumerate(modes) if value == mode]
            part = {k:v[indices] if isinstance(v, torch.Tensor) else [v[i] for i in indices] for k,v in batch.items()}
            scores[indices] = self.simulate(part, mode)['ranking_logit']
        return scores

    def load_checkpoint(self, checkpoint):
        if self.llm_actions and any(k in checkpoint['model_state_dict'] for k in ('diet_vlm.scale', 'exercise_vlm.scale')): raise ValueError('Factor-only action checkpoints require retraining with explicit and latent effectors')
        if 'metadata_encoder.mean' in checkpoint['model_state_dict']: raise ValueError('Numeric metadata checkpoints require retraining with semantic metadata')
        if checkpoint.get('ranking_version') != 2 and (checkpoint.get('phase') in ('professional','physiological') or checkpoint['model_state_dict'].get('ranked_modes', torch.zeros(3,dtype=torch.bool)).any()):
            raise ValueError('Bounded ranking checkpoints cannot resume DPO; restart alignment from a fusion checkpoint')
        if checkpoint.get('ranking_version') in (1,2): return self.load_state_dict(checkpoint['model_state_dict'])
        state = checkpoint['model_state_dict']
        missing = set(self.state_dict()) - set(state)
        allowed = {key for key in self.state_dict() if key.startswith('ranking_head.')} | {'ranked_modes'}
        if missing != allowed or set(state)-set(self.state_dict()): raise ValueError('Incompatible legacy numeric checkpoint')
        self.load_state_dict(state, strict=False)

    @torch.no_grad()
    def simulate(self, batch, mode):
        was_training = self.training
        self.eval()
        try: return self(batch, mode)
        finally: self.train(was_training)


class RankingHead(nn.Module):
    """Unbounded candidate logits for DPO and subsequent candidate-set softmax."""
    def __init__(self, dimension, heads, layers, architecture):
        super().__init__()
        self.projections = nn.ModuleDict({mode:nn.Linear(dimension*(2 if mode == 'joint' else 1), dimension) for mode in ('diet', 'exercise', 'joint')})
        self.backbone = transformer_encoder(dimension, heads, layers, architecture['dropout'],
            dimension*architecture['feedforward_multiplier'], architecture['activation'])
        self.score = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 1))

    def forward(self, trajectory, mode):
        tokens = self.projections[mode](trajectory)
        return self.score(self.backbone(tokens, mask=causal_mask(tokens.size(1), tokens.device)).mean(1)).squeeze(-1)
