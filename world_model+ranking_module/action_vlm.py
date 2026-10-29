import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class ActionVLM(nn.Module):
    """Qwen action factor extraction."""
    def __init__(self, path, domain):
        super().__init__()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if domain not in ('diet', 'exercise'): raise ValueError('Expected diet or exercise')
        self.domain = domain
        self.processor = AutoProcessor.from_pretrained(path)
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype='auto')
        self.factor_head = nn.Linear(self.vlm.config.text_config.hidden_size, 6 if domain == 'diet' else 3)
        artifact = Path(path) / 'action_factors.pt'
        self.backend_id = None
        if artifact.exists():
            state = torch.load(artifact, map_location='cpu', weights_only=True)
            if state['domain'] != domain: raise ValueError('Action encoder domain mismatch')
            self.factor_head.load_state_dict(state['factor_head']); self.backend_id = state['backend_id']

    def inputs(self, text, images=(), annotations=None):
        if not isinstance(text, str) or not isinstance(images, (list, tuple)): raise ValueError('Expected text and image list')
        if self.domain == 'exercise' and images: raise ValueError('Exercise encoder accepts text only')
        if not text.strip() and not images: raise ValueError('An action needs text or meal images')
        instruction = ('Describe food items, ingredients, preparation, portions and nutrition.' if self.domain == 'diet'
                       else 'Describe exercise type, intensity in MET, duration and timing relative to meals.')
        content = [{'type':'image', 'path':str(Path(p).resolve())} for p in images]
        content.append({'type':'text', 'text':instruction + '\n' + text})
        messages = [{'role':'user', 'content':content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        if annotations is None: return prompt, prompt['attention_mask'].clone()
        messages.append({'role':'assistant', 'content':[{'type':'text', 'text':json.dumps(annotations, ensure_ascii=False)}]})
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt')
        length = prompt['input_ids'].size(1)
        if not torch.equal(inputs['input_ids'][:, :length], prompt['input_ids']): raise ValueError('Annotation prompt prefix mismatch')
        labels = inputs['input_ids'].clone(); labels[:, :length] = -100
        inputs['labels'] = labels
        mask = inputs['attention_mask'].clone(); mask[:, length:] = 0
        return inputs, mask

    def forward(self, inputs, pooling_mask, factors=None):
        output = self.vlm(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = output.hidden_states[-1]
        mask = pooling_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        log_factors = F.softplus(self.factor_head(pooled.to(self.factor_head.weight.dtype)))
        result = {'factors':log_factors.expm1()}
        if factors is not None:
            valid = torch.isfinite(factors)
            if not valid.any() or (factors[valid] < 0).any(): raise ValueError('Nonnegative measured factor labels required')
            result['loss'] = F.mse_loss(log_factors[valid], factors[valid].log1p()) + output.loss
        return result

    def save(self, path, backend_id):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.vlm.save_pretrained(path); self.processor.save_pretrained(path)
        torch.save(dict(domain=self.domain, backend_id=backend_id, factor_head=self.factor_head.state_dict()), path/'action_factors.pt')

    @torch.inference_mode()
    def generate_factors(self, text, images=()):
        fields = 'carbohydrate_g, protein_g, fat_g, fiber_g, mass_g, energy_kcal' if self.domain == 'diet' else 'duration_min, energy_kcal, intensity_MET'
        text += '\nReturn only JSON {"factors":[numbers]} in this order: ' + fields + '. Use finite nonnegative numbers. Preserve supplied quantities; estimate unspecified quantities from the action. No prose.'
        inputs, _ = self.inputs(text, images)
        inputs = {k:v.to(self.vlm.device) for k,v in inputs.items()}
        generated = self.vlm.generate(**inputs, max_new_tokens=256, do_sample=False)
        answer = self.processor.batch_decode(generated[:, inputs['input_ids'].size(1):], skip_special_tokens=True)[0].strip()
        if answer.startswith('```'): answer = answer.split('\n',1)[1].rsplit('```',1)[0].strip()
        factors = json.loads(answer)['factors']
        if not isinstance(factors,list) or len(factors) != (6 if self.domain == 'diet' else 3): raise ValueError('Invalid generated action factor count')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in factors): raise ValueError('Generated action factors must be numbers')
        return torch.tensor(factors, device=self.vlm.device, dtype=torch.float32)
