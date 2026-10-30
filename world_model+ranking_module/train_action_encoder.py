import argparse
import json
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from action_vlm import ActionVLM
from configuration import settings, parse_configured, optimizer_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--domain', choices=['diet', 'exercise'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend-id', required=True)
    parser.add_argument('--epochs', type=int)
    args = parse_configured(parser, None, 'action_training', ('model', 'epochs'))
    training = settings('action_training')
    if args.epochs < 1: raise ValueError('Epochs must be positive')
    rows = json.loads(Path(args.data).read_text(encoding='utf-8'))
    if not rows: raise ValueError('Action supervision records required')
    schema = settings('schema', 'actions', args.domain)
    for row in rows:
        row['images'] = [str((Path(args.data).resolve().parent/p).resolve()) for p in row.get('images', [])]
        if len(row['factors']) != len(schema): raise ValueError('Factor label count mismatch')
        if not row.get('annotations'): raise ValueError('Structured annotation supervision required')
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model = ActionVLM(args.model, args.domain, schema).requires_grad_(True)
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], **optimizer_settings())
    loader = DataLoader(rows, batch_size=training['batch_size'], shuffle=True, collate_fn=list)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    for _ in range(args.epochs):
        for records in loader:
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for row in records:
                inputs, mask = accelerator.unwrap_model(model).inputs(row.get('text', ''), row['images'], row['annotations'])
                inputs = {k:v.to(accelerator.device) for k,v in inputs.items()}
                factors = torch.tensor([[float('nan') if v is None else v for v in row['factors']]], device=accelerator.device)
                loss = model(inputs, mask.to(accelerator.device), factors)['loss'] / len(records)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite action supervision loss')
                accelerator.backward(loss)
                total += float(loss.detach())
            accelerator.clip_grad_norm_(model.parameters(), training['gradient_clip']); optimizer.step()
            accelerator.print({'domain':args.domain, 'loss':total})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: accelerator.unwrap_model(model).save(args.output, args.backend_id)


if __name__ == '__main__': main()
