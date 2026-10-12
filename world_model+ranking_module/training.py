"""X-Life training objectives and phase-specific parameter policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


MODEL_INPUT_KEYS = {
    "metadata_input_ids", "metadata_attention_mask", "sleep_input_ids", "sleep_attention_mask",
    "food_input_ids", "food_attention_mask", "food_texts", "food_images",
    "exercise_input_ids", "exercise_attention_mask", "cgm_preprandial", "cgm_preprandial_mask"
}


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [to_device(value, device) for value in batch]
    return batch


def masked_mse(prediction, target, valid_mask):
    if prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("Prediction, observed CGM and validity mask must have equal shapes.")
    valid = valid_mask.bool() & torch.isfinite(target)
    if not valid.any():
        return prediction.sum() * 0.0
    return (prediction[valid] - target[valid]).square().mean()


def semantic_loss(prediction, target, valid_mask, mode="cosine"):
    if prediction.shape != target.shape or valid_mask.shape != target.shape[:1]:
        raise ValueError("Semantic targets must match the configured annotation dimension.")
    valid = valid_mask.bool() & torch.isfinite(target).all(dim=-1)
    if not valid.any():
        return prediction.sum() * 0.0
    if mode == "cosine":
        if (target[valid].norm(dim=-1) == 0).any():
            raise ValueError("Valid cosine supervision targets must have nonzero norm.")
        return (1.0 - F.cosine_similarity(prediction[valid], target[valid], dim=-1)).mean()
    if mode == "mse":
        return F.mse_loss(prediction[valid], target[valid])
    raise ValueError("semantic_loss must be cosine or mse.")


@dataclass
class LossWeights:
    diet_auxiliary: float = 0.25
    exercise_auxiliary: float = 0.25
    omics: float = 0.1
    cgm_alignment: float = 1.0
    beta: float = 0.1
    semantic_mode: str = "cosine"

    def __post_init__(self):
        for name in ("diet_auxiliary", "exercise_auxiliary", "omics", "cgm_alignment", "beta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "beta" and value == 0):
                raise ValueError(f"Invalid loss setting {name}={value}.")
        if self.semantic_mode not in {"cosine", "mse"}:
            raise ValueError("semantic_mode must be cosine or mse.")


def configure_training_phase(model, phase):
    """Freeze complete action encoders in Phase 2; update the ranker/G in Phase 3."""
    if phase not in {"encoder_pretraining", "world_model", "preference_alignment", "participant_update"}:
        raise ValueError(f"Unknown training phase {phase}.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if phase == "encoder_pretraining":
        names = ("image_encoder", "diet_action_projection", "exercise_action_projection",
                 "diet_annotation_head", "exercise_annotation_head")
        text_names = ("language_model", "exercise_projection")
        for name in text_names:
            for parameter in getattr(model.text_encoder, name).parameters():
                parameter.requires_grad_(True)
    elif phase == "world_model":
        names = ("cgm_encoder", "feature_fusion", "diet_transition", "exercise_transition",
                 "future_state_fusion", "ranking_head", "cgm_decoder", "diet_aux_decoder",
                 "exercise_aux_decoder", "omics_head")
        for name in ("metadata_projection", "sleep_projection"):
            for parameter in getattr(model.text_encoder, name).parameters():
                parameter.requires_grad_(True)
    else:
        # The glucose head G shares the ranker's latent representation. Its
        # CGM loss constrains the very parameters used for preference ranking.
        names = ("ranking_head", "cgm_decoder")
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            if phase in {"encoder_pretraining", "world_model"}:
                raise ValueError(f"Phase {phase} requires training-only module {name}.")
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.training_phase = phase
    set_training_mode(model)


def set_training_mode(model):
    """Frozen backbones must not add dropout noise to current/reference comparisons."""
    model.train()
    for module in (model.text_encoder.language_model, model.image_encoder):
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()
    if getattr(model, "training_phase", None) in {"preference_alignment", "participant_update"}:
        for name in ("text_encoder", "cgm_encoder", "feature_fusion", "diet_action_projection",
                     "exercise_action_projection", "diet_transition", "exercise_transition", "future_state_fusion"):
            getattr(model, name).eval()


def encoder_pretraining_objective(model, batch, weights):
    targets, masks = batch.get("semantic_targets", {}), batch.get("semantic_masks", {})
    if not any(key in masks and masks[key].any() for key in ("diet", "exercise")):
        raise ValueError("Encoder pretraining needs diet/exercise annotations encoded as semantic targets.")
    predictions = model.pretrain_annotations(**model_inputs(batch))
    losses = {}
    for key in ("diet", "exercise"):
        if key in targets:
            losses[key] = semantic_loss(predictions[key], targets[key], masks[key], weights.semantic_mode)
    total = sum(losses.values())
    return total, {key: value.detach() for key, value in losses.items()}


def world_model_objective(model, batch, weights):
    target = batch["cgm_postprandial"]
    mask = batch["cgm_postprandial_mask"].bool()
    if not mask.any():
        raise ValueError("World-model training needs measured future CGM targets.")
    details = model(**model_inputs(batch), cgm_postprandial=target,
                    return_details=True, compute_auxiliary=True)
    losses = {"cgm": masked_mse(details["predicted_cgm"], target, mask)}
    # Dual regularization supervises each branch decoder with the observed
    # CGM trajectory for its corresponding event type.
    for name, event in (("diet", "meal"), ("exercise", "exercise")):
        active = torch.tensor([kind == event for kind in batch["cgm_type"]], device=mask.device)
        losses[name + "_auxiliary"] = masked_mse(details[name + "_auxiliary_cgm"], target,
                                                   mask & active.unsqueeze(-1))
    targets, masks = batch.get("semantic_targets", {}), batch.get("semantic_masks", {})
    losses["omics"] = (semantic_loss(details["omics_prediction"], targets["omics"], masks["omics"],
                                      weights.semantic_mode) if "omics" in targets
                       else details["omics_prediction"].sum() * 0.0)
    total = (losses["cgm"] + weights.diet_auxiliary * losses["diet_auxiliary"]
             + weights.exercise_auxiliary * losses["exercise_auxiliary"] + weights.omics * losses["omics"])
    return total, {key: value.detach() for key, value in losses.items()}


class FrozenRankingReference(nn.Module):
    """Previous-stage score on the identical frozen physiological/action backbone."""
    def __init__(self, model):
        super().__init__()
        if any(parameter.requires_grad for module in (model.text_encoder, model.image_encoder, model.feature_fusion,
                                                       model.diet_action_projection, model.exercise_action_projection,
                                                       model.diet_transition, model.exercise_transition,
                                                       model.future_state_fusion, model.cgm_encoder)
               for parameter in module.parameters()):
            raise ValueError("Freeze the world-model backbone before capturing the ranking reference.")
        self.ranking_head = deepcopy(model.ranking_head).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, details):
        return self.ranking_head(details["fused_future_rollout"].detach(),
                                 details["physiological_state"].detach())[1]


def preference_objective(model, pairs, observed_batch, reference, weights, stage):
    if stage not in {"professional", "physiological", "user"}:
        raise ValueError("Alignment stage must be professional, physiological or user.")
    if reference is None:
        raise ValueError("Preference alignment requires a frozen previous-stage reference.")
    preferred = model(**model_inputs(pairs["preferred"]), return_details=True, decode_glucose=False)
    rejected = model(**model_inputs(pairs["rejected"]), return_details=True, decode_glucose=False)
    reference_margin = reference(preferred) - reference(rejected)
    margin = preferred["ranking_score"] - rejected["ranking_score"] - reference_margin.detach()
    per_pair = -F.logsigmoid(weights.beta * margin)
    pair_weights = pairs["pair_weight"].to(per_pair)
    if stage == "user":
        source = torch.tensor([value == "user" for value in pairs["source_stage"]], device=per_pair.device)
        count = pairs["selection_count"].to(per_pair).clamp_min(1.0)
        pair_weights = pair_weights * torch.where(source, count.rsqrt(), torch.ones_like(count))
        pair_weights = pair_weights * pairs["diversity_weight"].to(per_pair)
    if not torch.isfinite(pair_weights).all() or (pair_weights < 0).any() or pair_weights.sum() <= 0:
        raise ValueError("Preference weights must be finite, nonnegative and have a positive sum.")
    preference_loss = (per_pair * pair_weights).sum() / pair_weights.sum()
    # This is a separate factual event batch. The observed trajectory must
    # never be assigned to an unexecuted preferred/rejected alternative.
    if observed_batch is None and weights.cgm_alignment > 0:
        raise ValueError("Supply factual observed events for the CGM alignment regularizer.")
    if weights.cgm_alignment > 0:
        target = observed_batch["cgm_postprandial"]
        mask = observed_batch["cgm_postprandial_mask"]
        if not mask.any():
            raise ValueError("CGM alignment regularization needs measured targets.")
        predicted = model(**model_inputs(observed_batch), cgm_postprandial=target)
        cgm_loss = masked_mse(predicted, target, mask)
    else:
        cgm_loss = preference_loss * 0.0
    return preference_loss + weights.cgm_alignment * cgm_loss, {
        "preference": preference_loss.detach(), "cgm": cgm_loss.detach(),
        "pair_weight_sum": pair_weights.sum().detach()}
