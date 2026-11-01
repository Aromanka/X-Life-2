import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world_model import MetabolicWorldModel


class EchoDecoder(nn.Module):
    def forward(self, encoded_cgm, memory):
        return memory[:, :, :1]


class TransitionAlignmentChecks(unittest.TestCase):
    def model(self, readout='direct', echo=False):
        with patch.dict(os.environ, XLIFE_USE_LLM_ACTION_ENCODERS='0'):
            model = MetabolicWorldModel(dimension=4, basis_dim=2, heads=2, layers=1,
                                        patch_size=2, action_backend_id='fixture', state_pooling='both',
                                        gap_feature=True, readout=readout, dropout=0., feedforward=8,
                                        feedforward_multiplier=2, type_capacity=4, position_max_length=16,
                                        time_encoding_base=10000., activation='gelu',
                                        data_config=dict(horizon_steps=4, interval_minutes=5)).eval()
        if readout == 'autoregressive' and echo:
            for mode in ('diet', 'exercise', 'joint'): model.ar_decoders[mode] = EchoDecoder()
        return model

    def test_all_branches_current_state_readout(self):
        for readout in ('direct', 'autoregressive'):
            model = self.model(readout, echo=True)
            with torch.no_grad():
                model.gate.weight.zero_(); model.gate.bias.zero_()
            for mode in ('diet', 'exercise', 'joint'):
                for horizon in (1, 4):
                    with self.subTest(mode=mode, horizon=horizon, readout=readout):
                        diet = torch.tensor([[1., 2., 3., 4.]])
                        exercise = torch.tensor([[5., 6., 7., 8.]])
                        initial = torch.cat([diet, exercise], -1) if mode == 'joint' else diet if mode == 'diet' else exercise
                        current = torch.stack([initial*2**i for i in range(horizon)], 1)
                        expected = current
                        transition = model.fusion if mode == 'joint' else getattr(model, mode)
                        with patch.object(model, 'initial_states', return_value=(diet, exercise, torch.tensor([5.]))), \
                             patch.object(model.diet, 'increment', side_effect=lambda s:s) as diet_step, \
                             patch.object(model.exercise, 'increment', side_effect=lambda s:s) as exercise_step, \
                             patch.object(model.fusion, 'increment', side_effect=lambda s:s) as joint_step, \
                             patch.object(transition, 'readout', side_effect=lambda s:(s, s[:, :, 0])) as readout_call:
                            output = model({}, mode, horizon=horizon)
                        torch.testing.assert_close(readout_call.call_args.args[0], expected)
                        torch.testing.assert_close(output['states'], expected)
                        torch.testing.assert_close(output['trajectory'], expected)
                        torch.testing.assert_close(output['predicted_cgm'], 5.+expected[:, :, 0])
                        steps = [joint_step] if mode == 'joint' else [diet_step if mode == 'diet' else exercise_step]
                        for step in steps:
                            self.assertEqual(step.call_count, horizon)
                            for i, call in enumerate(step.call_args_list):
                                torch.testing.assert_close(call.args[0], current[:, i])
                        if mode == 'joint':
                            torch.testing.assert_close(output['gates'], torch.full((1, horizon, 3), .5))
                            for i in range(horizon):
                                torch.testing.assert_close(diet_step.call_args_list[i].args[0], current[:, i, :4])
                                torch.testing.assert_close(exercise_step.call_args_list[i].args[0], current[:, i, 4:])

    def test_first_output_is_independent_of_current_transition(self):
        for readout in ('direct', 'autoregressive'):
            model = self.model(readout)
            for mode in ('diet', 'exercise', 'joint'):
                with self.subTest(mode=mode, readout=readout):
                    diet = torch.tensor([[1., -2., 3., -.5]])
                    exercise = torch.tensor([[-3., 1., .5, 4.]])
                    with patch.object(model, 'initial_states', return_value=(diet, exercise, torch.tensor([5.]))):
                        baseline = model({}, mode, horizon=3)
                        transition = model.fusion if mode == 'joint' else getattr(model, mode)
                        with patch.object(transition, 'increment', side_effect=lambda s:torch.arange(s.size(-1), dtype=s.dtype).expand_as(s)):
                            changed = model({}, mode, horizon=3)
                    torch.testing.assert_close(baseline['trajectory'][:, 0], changed['trajectory'][:, 0])
                    torch.testing.assert_close(baseline['predicted_cgm'][:, 0], changed['predicted_cgm'][:, 0])
                    self.assertFalse(torch.allclose(baseline['trajectory'][:, 1:], changed['trajectory'][:, 1:]))

    def test_single_readout_uses_basis_and_joint_readout_uses_full_state(self):
        model = self.model()
        for transition, dimension in ((model.diet, 4), (model.exercise, 4), (model.fusion, 8)):
            with self.subTest(dimension=dimension, transition=transition):
                states = torch.randn(1, 3, dimension)
                baseline, glucose = transition.readout(states)
                glucose.sum().backward()
                if transition is model.fusion:
                    self.assertIsNone(transition.basis.grad)
                else:
                    self.assertIsNotNone(transition.basis.grad)
                    self.assertGreater(float(transition.basis.grad.abs().sum()), 0.)
                with torch.no_grad(): transition.basis[0].neg_()
                changed, _ = transition.readout(states)
                self.assertEqual(baseline.shape, states.shape)
                if transition is model.fusion:
                    torch.testing.assert_close(baseline, changed)
                else:
                    self.assertFalse(torch.allclose(baseline, changed))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
