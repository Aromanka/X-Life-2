import json
import logging
from collections.abc import Mapping
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from action_factor_format import parse_action_factors


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
        if annotations is None:
            instruction = 'Extract physiological action factors. Follow the requested JSON schema exactly.'
        content = [{'type':'image', 'path':str(Path(p).resolve())} for p in images]
        content.append({'type':'text', 'text':instruction + '\n' + text})
        messages = [{'role':'user', 'content':content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        self.validate_processor_output(prompt)
        if annotations is None: return prompt, prompt['attention_mask'].clone()
        messages.append({'role':'assistant', 'content':[{'type':'text', 'text':json.dumps(annotations, ensure_ascii=False)}]})
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt')
        self.validate_processor_output(inputs)
        length = prompt['input_ids'].size(1)
        if not torch.equal(inputs['input_ids'][:, :length], prompt['input_ids']): raise ValueError('Annotation prompt prefix mismatch')
        labels = inputs['input_ids'].clone(); labels[:, :length] = -100
        inputs['labels'] = labels
        mask = inputs['attention_mask'].clone(); mask[:, length:] = 0
        return inputs, mask

    @staticmethod
    def validate_processor_output(inputs):
        if not isinstance(inputs, Mapping) or not {'input_ids', 'attention_mask'} <= inputs.keys():
            raise ValueError('Qwen processor must return input_ids and attention_mask in a mapping; '
                             f'got {type(inputs).__name__}. Check the model processor and transformers version '
                             '(requirements.txt pins transformers==4.57.1).')

    def parse_factors(self, answer):
        return parse_action_factors(answer, self.domain)

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
        count = 6 if self.domain == 'diet' else 3
        request = (text + '\nReturn ONLY one JSON object with key "factors" containing exactly '
                   + str(count) + ' numeric values in this order: ' + fields
                   + '. Use finite nonnegative numbers, not strings or null. Preserve supplied quantities; '
                   'estimate unspecified quantities from the action. Output totals for the entire action. '
                   'Multiple images may show the same meal; do not double count. '
                   'No per-food lists, alternative answers, markdown or prose.')
        errors, answer = [], ''
        for attempt in range(3):
            correction = '' if not errors else ('\nYour previous response was invalid: ' + errors[-1]
                + '\nPrevious response:\n' + answer[:2000]
                + f'\nCorrect the format and provide ALL {count} factors, including missing estimates. Return one aggregate vector.')
            inputs, _ = self.inputs(request + correction, images)
            inputs = {k:v.to(self.vlm.device) for k,v in inputs.items()}
            generated = self.vlm.generate(**inputs, max_new_tokens=512, do_sample=False)
            answer = self.processor.batch_decode(generated[:, inputs['input_ids'].size(1):], skip_special_tokens=True)[0].strip()
            try:
                factors = self.parse_factors(answer)
                vector = torch.tensor(factors, device=self.vlm.device, dtype=torch.float32)
                if not torch.isfinite(vector).all():
                    raise ValueError('Action factors exceed float32 range')
                self.factor_generation_audit = dict(attempts=attempt+1, validation_errors=errors)
                return vector
            except ValueError as error:
                errors.append(str(error))
                logging.getLogger(__name__).warning('%s action format attempt %d/3: %s; response=%r',
                    self.domain, attempt+1, error, answer[:1000])
        raise ValueError(f'{self.domain} action factors invalid after 3 attempts: {errors[-1]}; '
                         f'model_response={answer[:1000]!r}')
