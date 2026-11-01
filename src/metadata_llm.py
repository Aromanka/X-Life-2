import json
import torch


class MetadataLLM:
    """Frozen metadata semantic features."""
    def __init__(self, path, device):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model_class = Qwen2_5_VLForConditionalGeneration if AutoConfig.from_pretrained(path).model_type == 'qwen2_5_vl' else AutoModelForCausalLM
        self.model = model_class.from_pretrained(path, torch_dtype='auto').to(device).eval().requires_grad_(False)
        self.backend_id = str(path)

    @torch.inference_mode()
    def encode(self, fields):
        messages = [{'role':'system','content':'Represent the supplied clinical and physiological metadata. Preserve field identifiers, units, missing values and categorical findings. Do not infer clinical meanings for unnamed array positions.'},
                    {'role':'user','content':json.dumps({'fields':fields}, ensure_ascii=False, allow_nan=False)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        hidden = self.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        vector = ((hidden.float()*mask).sum(1)/mask.sum(1))[0]
        if not torch.isfinite(vector).all(): raise ValueError('Nonfinite metadata embedding')
        return {'embedding':vector.cpu().tolist(), 'backend_id':self.backend_id}
