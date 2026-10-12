"""Cache local Qwen3 embeddings of structured training annotations.

Annotations are standardized and encoded as semantic targets for the action
encoders and omics head. Targets support cosine-distance or MSE supervision.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModel
from dataset import get_tokenizer
from model import local_model_directory, hidden_size, masked_token_mean
from train_data import event_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-path', required=True, help='Existing local Qwen3 teacher directory.')
    parser.add_argument('--image-dir')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-length', type=int, default=1024)
    args = parser.parse_args(argv)
    records = event_records(args.input, args.image_dir)
    path = local_model_directory(args.model_path, 'model-path')
    tokenizer = get_tokenizer(model_path=path)
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True, torch_dtype='auto')
    teacher = path
    model.to(args.device).eval()
    dimension, counts = hidden_size(model.config), dict(diet=0, exercise=0, omics=0)
    with torch.inference_mode():
        for record in records:
            targets = record.setdefault('training_targets', {})
            for kind in counts:
                annotation = record.get('training_annotations', {}).get(kind)
                if annotation is None:
                    continue
                if not isinstance(annotation, (dict, str, list)) or not annotation:
                    raise ValueError(f'{kind} annotations must be nonempty structured labels or text.')
                text = json.dumps(annotation, ensure_ascii=False, sort_keys=True)
                tokens = tokenizer(text, max_length=args.max_length, truncation=True, padding=True, return_tensors='pt')
                tokens = {key: value.to(args.device) for key, value in tokens.items()}
                hidden = model(**tokens, use_cache=False).last_hidden_state
                vector = masked_token_mean(hidden, tokens['attention_mask'])[0].float().cpu()
                if not torch.isfinite(vector).all() or vector.norm() == 0:
                    raise ValueError('Annotation encoder returned an invalid semantic target.')
                targets[kind] = vector.tolist()
                counts[kind] += 1
    if not sum(counts.values()):
        raise ValueError('Input must contain nonempty training_annotations.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {'teacher': teacher, 'annotation_dim': dimension, 'counts': counts,
                'pooling': 'attention-mask mean hidden state'}
    output.with_suffix(output.suffix + '.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata))
    return records


if __name__ == '__main__':
    main()
