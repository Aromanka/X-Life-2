"""Frozen Qwen3 embedding encoder for knowledge retrieval.

Follows https://huggingface.co/Qwen/Qwen3-Embedding-8B:
query instructions, last non-padding token pooling and L2 normalization.
"""
import torch
from torch.nn import functional as F


class KnowledgeEncoder:
    def __init__(self, path, device, max_length, query_instruction):
        from transformers import AutoModel, AutoTokenizer
        if type(max_length) is not int or max_length < 1: raise ValueError('Positive embedding_max_length required')
        if not isinstance(query_instruction, str) or not query_instruction.strip(): raise ValueError('Embedding query instruction required')
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.model = AutoModel.from_pretrained(path, torch_dtype='auto').to(device).eval().requires_grad_(False)
        self.dimension = self.model.config.hidden_size
        self.backend_id = str(path)
        self.revision = getattr(self.model.config, '_commit_hash', None)
        self.max_length, self.query_instruction = max_length, query_instruction

    @staticmethod
    def pool(hidden, attention_mask):
        positions = torch.arange(attention_mask.size(1), device=hidden.device).expand_as(attention_mask)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if (last < 0).any(): raise ValueError('Embedding input has no valid tokens')
        return hidden[torch.arange(len(hidden), device=hidden.device), last]

    @torch.inference_mode()
    def encode(self, text, input_type, dimension):
        if not isinstance(text, str) or not text.strip(): raise ValueError('Embedding text must be nonempty')
        if input_type not in ('query', 'document'): raise ValueError('Embedding input_type must be query or document')
        if dimension != self.dimension: raise ValueError(f'Embedding model outputs {self.dimension} dimensions; configure both knowledge.dimension and retrieval dimension to match')
        if input_type == 'query': text = f'Instruct: {self.query_instruction}\nQuery:{text}'
        inputs = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs = {key:value.to(self.model.device) for key,value in inputs.items()}
        output = self.model(**inputs, use_cache=False, return_dict=True)
        vector = self.pool(output.last_hidden_state, inputs['attention_mask']).float()
        if not torch.isfinite(vector).all() or (vector.norm(dim=-1) <= 0).any(): raise ValueError('Invalid knowledge embedding')
        return F.normalize(vector, p=2, dim=-1)[0].cpu().tolist()
