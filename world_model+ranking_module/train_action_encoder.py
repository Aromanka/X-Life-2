import argparse
import json
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from action_vlm import ActionVLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-72B-Instruct')
    parser.add_argument('--domain', choices=['diet', 'exercise'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend-id', required=True)
    parser.add_argument('--schema', default=str(Path(__file__).resolve().parents[2]/'extra_code/schema.json'))
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1: raise ValueError('Epochs must be positive')
    rows = json.loads(Path(args.data).read_text(encoding='utf-8'))
    if not rows: raise ValueError('Action supervision records required')
    schema = json.loads(Path(args.schema).read_text(encoding='utf-8'))['actions'][args.domain]
    for row in rows:
        row['images'] = [str((Path(args.data).resolve().parent/p).resolve()) for p in row.get('images', [])]
        if len(row['factors']) != len(schema): raise ValueError('Factor label count mismatch')
        if not row.get('annotations'): raise ValueError('Structured annotation supervision required')
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model = ActionVLM(args.model, args.domain, schema).requires_grad_(True)
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loader = DataLoader(rows, batch_size=1, shuffle=True, collate_fn=list)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    for _ in range(args.epochs):
        for records in loader:
            row = records[0]
            inputs, mask = accelerator.unwrap_model(model).inputs(row.get('text', ''), row['images'], row['annotations'])
            inputs = {k:v.to(accelerator.device) for k,v in inputs.items()}
            factors = torch.tensor([[float('nan') if v is None else v for v in row['factors']]], device=accelerator.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs, mask.to(accelerator.device), factors)['loss']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite action supervision loss')
            accelerator.backward(loss); accelerator.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            accelerator.print({'domain':args.domain, 'loss':float(loss.detach())})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: accelerator.unwrap_model(model).save(args.output, args.backend_id)


if __name__ == '__main__': main()
