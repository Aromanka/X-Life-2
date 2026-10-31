import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'extra_code'), str(ROOT/'X-Life-2/world_model+ranking_module')]
from action_inputs import action_inputs
from alignment import candidate, batch_candidates
from services.contracts import validate_candidate
from train import collate
from world_model import MetabolicWorldModel


class ActionInputChecks(unittest.TestCase):
    def action(self, mode):
        return dict(mode=mode, diet=[40.,20.,10.,5.,300.,400.], exercise=[20.,70.,3.],
                    exercise_type='cycling', intensity_scale='MET', onset=[0.,0. if mode == 'diet' else -15.],
                    description='Meal and cycling.',
                    units=dict(diet=['g']*5+['kcal'], exercise=['min','kcal','scale'], onset='min'))

    def requests(self, actions, mode, training=False):
        data = dict(history_steps=6, horizon_steps=2, interval_minutes=5)
        metadata = dict(fields=[], embedding=[.1,.2], backend_id='fixture', test_backend=True)
        context = dict(subject_id='fixture', origin='2026-10-31T12:00:00', history=[5.]*6,
                       history_mask=[True]*6, metadata=metadata)
        requests = []
        def respond(backend, role, payload):
            self.assertEqual(role, 'action/factors')
            record = json.loads(payload['text'])
            requests.append((payload['domain'], record, payload['images']))
            values = record[payload['domain']]
            return dict(result=dict(backend_id='fixture', factors=values, latent=[.1,.2],
                factor_schema=[dict(name=str(i), unit=u, scale=1.) for i,u in enumerate(record['units'])]),
                provenance=dict(test_backend=True))
        with patch('configuration._config', dict(data=data, http=dict(url='http://fixture', timeout=1))), \
             patch.dict(os.environ, XLIFE_USE_LLM_ACTION_ENCODERS='1'), \
             patch('alignment.prepare_metadata', return_value=metadata), \
             patch('backends.action_encoder.HTTPBackend.call', respond):
            samples = [candidate(context, validate_candidate(action), {}) for action in actions]
            if training:
                for sample in samples:
                    sample.update(subject_id=context['subject_id'], origin=context['origin'],
                                  target=torch.zeros(2), target_mask=torch.ones(2, dtype=torch.bool))
                batch = collate(samples, {})
            else:
                batch = batch_candidates(samples, 'cpu')
            model = MetabolicWorldModel(dimension=8, basis_dim=4, heads=2, layers=1, patch_size=2,
                action_backend_id='fixture', state_pooling='both', gap_feature=True, readout='direct',
                dropout=0., feedforward=16, feedforward_multiplier=2, type_capacity=4,
                position_max_length=16, time_encoding_base=10000., activation='gelu', data_config=data).eval()
            with torch.no_grad(): model(batch, mode)
        return requests

    def test_quantities_reach_vlm_despite_identical_descriptions(self):
        for mode in ('diet', 'exercise', 'joint'):
            with self.subTest(mode=mode):
                first = self.action(mode)
                second = deepcopy(first)
                second['diet'] = [v*2 for v in first['diet']]
                second['exercise'] = [60.,210.,6.]
                requests = self.requests([first, second], mode)
                for domain in (('diet','exercise') if mode == 'joint' else (mode,)):
                    records = [record for name,record,_ in requests if name == domain]
                    self.assertEqual([record[domain] for record in records], [first[domain], second[domain]])
                    for record in records:
                        self.assertEqual(record['description'], first['description'])
                        self.assertEqual(record['units'], first['units'][domain])
                        self.assertEqual(record['onset_min'], first['onset'])

    def test_named_intensity_scales_are_preserved(self):
        for scale in ('MET', 'Borg6_20', 'Borg0_10'):
            with self.subTest(scale=scale):
                action = self.action('exercise')
                action['intensity_scale'] = scale
                record = self.requests([action], 'exercise')[0][1]
                self.assertEqual(record['exercise'], action['exercise'])
                self.assertEqual(record['exercise_type'], 'cycling')
                self.assertEqual(record['intensity_scale'], scale)

    def test_training_and_preference_batches_send_same_inputs(self):
        for mode in ('diet', 'exercise', 'joint'):
            with self.subTest(mode=mode):
                action = self.action(mode)
                self.assertEqual(self.requests([action], mode), self.requests([action], mode, training=True))

    def test_images_text_and_missing_inputs(self):
        image = action_inputs(dict(mode='diet', diet_images=['meal.jpg']))
        self.assertEqual(image['diet_images'], ['meal.jpg'])
        self.assertEqual(json.loads(image['diet_text'])['diet'], [None]*6)
        text = action_inputs(dict(mode='exercise', exercise_text='Walk gently.'))
        self.assertEqual(json.loads(text['exercise_text'])['description'], 'Walk gently.')
        self.assertEqual(text['diet_text'], '')
        for mode in ('diet', 'exercise', 'joint'):
            with self.assertRaisesRegex(ValueError, 'active action requires'):
                action_inputs(dict(mode=mode))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
