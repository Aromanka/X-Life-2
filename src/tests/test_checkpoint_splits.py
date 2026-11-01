from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'extra_code'), str(ROOT/'X-Life-2/src')]
from services.data_processing import split_samples
from train import main
from world_model import MetabolicWorldModel


class DatasetFixture:
    def __init__(self, samples): self.samples, self.skipped = samples, 0
    def __len__(self): return len(self.samples)


class CheckpointSplitChecks(unittest.TestCase):
    def test_initialization_uses_config_and_resume_and_prediction_restore_splits(self):
        model_config = dict(dimension=8, basis_dim=4, heads=2, layers=1, patch_size=2,
            action_backend_id='fixture', state_pooling='both', gap_feature=True, readout='direct',
            dropout=0., feedforward=16, feedforward_multiplier=2, type_capacity=4,
            position_max_length=16, time_encoding_base=10000., activation='gelu')
        data = dict(history_steps=6, horizon_steps=4, interval_minutes=5, train_fraction=.8,
                    validation_fraction=.1, within_subject_train_fraction=.7, metric_horizons_minutes=[20])
        training = dict(dataset='xlife_cleaned', phase='single', domain='diet', split='test',
            split_strategy='subject', epochs=1, batch_size=16, device='cpu', seed=42,
            rank_only=False, predict_only=False, joint_only=False, learning_rate=.001, gradient_clip=1.,
            scheduler=dict(factor=.5, patience=2, min_lr=.000001), early_stop_patience=3,
            optimizer=dict(betas=[.9,.999], eps=1e-8, weight_decay=.01))
        schema = dict(supervision=[dict(name='cgm', shape=[4], readout='sequence', prediction='predicted_cgm',
            path=['sample','target'], mask=['sample','target_mask'], loss='mse', weight=1., scale='glucose_scale')])
        samples = []
        for i in range(20):
            for day in (1,2):
                target = torch.tensor([5.1,5.2,5.3,5.4])
                mask = torch.ones(4, dtype=torch.bool)
                samples.append(dict(subject_id=f'new_{i}', origin=f'2026-01-{day:02d}T12:00:00',
                    mode='diet', joint=False, diet=torch.ones(6), exercise=torch.full((3,),float('nan')),
                    onset=torch.zeros(2), history=torch.arange(6)*.1+5., history_mask=torch.ones(6,dtype=torch.bool),
                    target=target, target_mask=mask, exercise_type='', diet_text='', exercise_text='',
                    metadata=dict(fields=[], embedding=[.1,.2], backend_id='fixture', test_backend=True),
                    supervision=dict(cgm=dict(target=target, mask=mask, test_backend=True))))
        config = dict(model=model_config, data=data, training=training)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, XLIFE_USE_LLM_ACTION_ENCODERS='0'), \
             patch('configuration._config', None), patch('configuration._config_path', None), \
             patch('train.load_schema', return_value=schema), \
             patch('train.XLifeCleanedDataset', return_value=DatasetFixture(samples)), \
             patch('train.split_samples', side_effect=split_samples), redirect_stdout(io.StringIO()):
            root = Path(directory)
            path = root/'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            old_data = dict(data, train_fraction=.5, validation_fraction=.25, within_subject_train_fraction=.5)
            old_model_config = dict(model_config, data_config=old_data, schema=schema)
            model = MetabolicWorldModel(**old_model_config)
            model.fit(samples)
            old = root/'old.pt'
            torch.save(dict(architecture_version=3, ranking_version=2, phase='single', llm_actions=False,
                model_config=old_model_config, model_state_dict=model.state_dict(), vocabulary={},
                split_strategy='within_subject', splits=dict(train=['old_train'], validation=['old_validation'], test=['old_test']),
                trained_domains=['diet']), old)
            output = root/'fine_tuned'
            main(['--data','fixture','--config',str(path),'--init-checkpoint',str(old),
                  '--output',str(output),'--max-steps','1'])
            saved = torch.load(output/'checkpoint.pt', weights_only=False)
            self.assertEqual(saved['split_strategy'], 'subject')
            self.assertEqual({key:len(value) for key,value in saved['splits'].items()}, dict(train=16,validation=2,test=2))
            self.assertEqual(set(sum(saved['splits'].values(), [])), {f'new_{i}' for i in range(20)})
            self.assertEqual(saved['model_config']['data_config'], data)
            self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], training['learning_rate'])
            config['training'].update(split_strategy='within_subject', seed=99)
            path.write_text(json.dumps(config), encoding='utf-8')
            resumed = root/'resumed'
            main(['--data','fixture','--config',str(path),'--resume',str(output/'checkpoint.pt'),
                  '--output',str(resumed),'--max-steps','1'])
            continued = torch.load(resumed/'checkpoint.pt', weights_only=False)
            self.assertEqual(continued['splits'], saved['splits'])
            self.assertEqual(continued['split_strategy'], 'subject')
            prediction = root/'prediction'
            main(['--data','fixture','--config',str(path),'--init-checkpoint',str(output/'checkpoint.pt'),
                  '--predict-only','--output',str(prediction)])
            rows = json.loads((prediction/'predictions.json').read_text(encoding='utf-8'))
            self.assertEqual({row['subject_id'] for row in rows}, set(saved['splits']['test']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
