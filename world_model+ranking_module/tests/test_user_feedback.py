"""Deployment regression tests with real model gradients and checkpoint resume."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import configure, settings
from alignment import candidate, FrozenRankingReference, preference_objective
from train_user_feedback import main, observation_records, observation_objective
from world_model import MetabolicWorldModel


class FeedbackChecks(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, XLIFE_USE_LLM_ACTION_ENCODERS='0')
        self.env.start()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        config = configure()
        config['data'].update(history_steps=6, horizon_steps=4)
        self.config_path = self.directory/'config.json'
        self.config_path.write_text(json.dumps(config), encoding='utf-8')
        configure(self.config_path)
        self.schema = dict(supervision=[dict(name='cgm', shape=[4], prediction='predicted_cgm', readout='sequence',
                                           loss='mse', weight=1., scale='glucose_scale')])
        self.context = dict(subject_id='p1', origin='2026-10-30T08:00:00', history=[5.]*6, history_mask=[True]*6,
                            metadata=dict(fields=[], embedding=[1., 2., 3.], backend_id='fixture', test_backend=True))
        self.actions = {mode:dict(mode=mode, diet=[40., 20., 10., 5., 300., 400.], exercise=[20., 3., 60.], onset=[0., 0.])
                        for mode in ('diet', 'exercise', 'joint')}
        self.observations = [dict(observation_id=mode, state_id='state', split='train', context=self.context,
                                 action=action, target=[6., None, 7., 8.], target_mask=[True, False, True, True])
                             for mode, action in self.actions.items()]

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()
        configure()

    def model(self, readout):
        config = dict(dimension=8, basis_dim=4, heads=2, layers=1, patch_size=2, feedforward=16,
                      readout=readout, schema=self.schema, data_config=settings('data'))
        model = MetabolicWorldModel(**config).eval()
        model.metadata_encoder([self.context['metadata']])
        return model, config

    def test_only_glucose_heads_receive_physiological_gradients(self):
        rows = observation_records(self.observations, {}, self.schema, 4)['train']
        for readout in ('direct', 'autoregressive'):
            with self.subTest(readout=readout):
                model, _ = self.model(readout)
                model.configure_phase('user_feedback')
                loss = observation_objective(model, rows, 'cpu')
                loss.backward()
                head_ids = {id(p) for mode in self.actions for p in model.glucose_simulation_head(mode).parameters()}
                for name, parameter in model.named_parameters():
                    if id(parameter) in head_ids:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                changed = deepcopy(rows)
                changed[0]['supervision']['cgm']['target'][1] = 100000.
                torch.testing.assert_close(loss, observation_objective(model, changed, 'cpu'))

    def test_dpo_does_not_update_glucose_heads(self):
        model, _ = self.model('direct')
        model.configure_phase('physiological')
        reference = FrozenRankingReference(model)
        model.configure_phase('user_feedback')
        pairs = [dict(preferred=candidate(self.context, self.actions['diet'], {}),
                      rejected=candidate(self.context, self.actions['exercise'], {}),
                      preferred_mode='diet', rejected_mode='exercise', pair_weight=1.)]
        loss, _ = preference_objective(model, pairs, reference, 2., 'cpu')
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.ranking_head.parameters()))
        for mode in self.actions:
            self.assertTrue(all(p.grad is None for p in model.glucose_simulation_head(mode).parameters()))

    def test_invalid_measurements_and_duplicate_observations(self):
        rows = observation_records(self.observations*2, {}, self.schema, 4)
        self.assertEqual(len(rows['train']), 3)
        for mutation in (dict(target_mask=[False]*4), dict(target=[None]*4), dict(target=[1.]),
                         dict(target_mask=[1, 2, 1, 1]), dict(observation_id='')):
            invalid = dict(self.observations[0], **mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                observation_records([invalid], {}, self.schema, 4)
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            observation_records([self.observations[0], dict(self.observations[0], split='validation')], {}, self.schema, 4)

    def test_cli_updates_both_heads_and_resume_matches_uninterrupted(self):
        for readout in ('direct', 'autoregressive'):
            with self.subTest(readout=readout):
                model, config = self.model(readout)
                initial = deepcopy(model.state_dict())
                parent = self.directory/'initial.pt'
                torch.save(dict(architecture_version=3, ranking_version=2, phase='physiological', llm_actions=False,
                                model_config=config, model_state_dict=initial, vocabulary={}), parent)
                common = dict(state_id='state', split='train', context=self.context,
                              preferred=self.actions['diet'], rejected=self.actions['exercise'])
                feedback = dict(common, source_stage='user_feedback', feedback_id='f1', preferred_id='d1', rejected_id='e1')
                expert = dict(common, source_stage='professional')
                physiological = dict(common, source_stage='physiological', evidence=dict(source='fixture',
                                     preferred_observation_id='a', rejected_observation_id='b'))
                observed = self.observations + [dict(self.observations[0], observation_id='validation', state_id='validation',
                                                    split='validation', context=dict(self.context, origin='2026-10-31T08:00:00'))]
                paths = {}
                for name, records in dict(data=[feedback], expert_replay_data=[expert], physiological_replay_data=[physiological], observation_data=observed).items():
                    paths[name] = self.directory/(name+'.json')
                    paths[name].write_text(json.dumps(records), encoding='utf-8')
                args = ['--config', str(self.config_path), '--batch-size', '8', '--learning-rate', '0.001']
                for key, value in paths.items(): args += ['--'+key.replace('_', '-'), str(value)]
                output = self.directory/readout
                main(args+['--init-checkpoint', str(parent), '--output', str(output), '--epochs', '1'])
                first = torch.load(output/'checkpoint.pt', weights_only=False)
                for mode in self.actions:
                    prefix = ('ar_decoders.'+mode+'.') if readout == 'autoregressive' else (('fusion' if mode == 'joint' else mode)+'.decoder.')
                    self.assertTrue(any(not torch.equal(initial[k], first['model_state_dict'][k]) for k in initial if k.startswith(prefix)))
                self.assertTrue(any(not torch.equal(initial[k], first['model_state_dict'][k]) for k in initial if k.startswith('ranking_head.')))
                allowed = {id(p) for p in model.ranking_head.parameters()} | {id(p) for m in self.actions for p in model.glucose_simulation_head(m).parameters()}
                for name, p in model.named_parameters():
                    if id(p) not in allowed: torch.testing.assert_close(initial[name], first['model_state_dict'][name], rtol=0, atol=0)
                self.assertEqual(first['user_feedback']['history'][0]['observations'], 3)
                self.assertIsNotNone(first['user_feedback']['history'][0]['validation_physiological_loss'])
                main(args+['--resume', str(output/'checkpoint.pt'), '--output', str(output), '--epochs', '1'])
                resumed = torch.load(output/'checkpoint.pt', weights_only=False)
                full_output = self.directory/(readout+'-full')
                main(args+['--init-checkpoint', str(parent), '--output', str(full_output), '--epochs', '2'])
                full = torch.load(full_output/'checkpoint.pt', weights_only=False)
                for key, value in full['model_state_dict'].items():
                    if isinstance(value, torch.Tensor): torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
                self.assertEqual(full['user_feedback']['history'], resumed['user_feedback']['history'])
                if readout == 'direct':
                    longer = observed + [dict(self.observations[0], observation_id='extra-'+str(i)) for i in range(2)]
                    paths['observation_data'].write_text(json.dumps(longer), encoding='utf-8')
                    unequal_output = self.directory/'unequal'
                    main(args+['--init-checkpoint', str(parent), '--output', str(unequal_output), '--epochs', '1', '--batch-size', '1'])
                    unequal = torch.load(unequal_output/'checkpoint.pt', weights_only=False)['user_feedback']
                    self.assertEqual(unequal['global_step'], 5)
                    self.assertEqual(unequal['history'][0]['observations'], 5)
                    self.assertEqual(unequal['history'][0]['pairs'], 3)
                    limited_output = self.directory/'limited'
                    main(args+['--init-checkpoint', str(parent), '--output', str(limited_output), '--epochs', '1', '--batch-size', '1', '--max-steps', '1'])
                    limited = torch.load(limited_output/'checkpoint.pt', weights_only=False)['user_feedback']
                    self.assertEqual(limited['global_step'], 1)
                    self.assertEqual(limited['history'][0]['observations'], 1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
