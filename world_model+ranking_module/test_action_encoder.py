import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from test_metadata import encoded_metadata
from torch import nn
from action_vlm import ActionVLM
from backends.action_encoder import HTTPActionEncoder
from world_model import MetabolicWorldModel
from alignment import candidate, batch_candidates, FrozenRankingReference, preference_objective


class ActionEncoderTests(unittest.TestCase):
    def response(self, role, payload):
        size = 6 if payload['domain'] == 'diet' else 3
        return {'result':{'factors':[2.]*size, 'backend_id':'sft', 'intensity_scale':'MET'}, 'provenance':{'test_backend':False}}

    def test_factor_bottleneck_and_images(self):
        encoder = HTTPActionEncoder('diet', 8, 'sft')
        with patch('backends.action_encoder.HTTPBackend.call', side_effect=self.response) as call:
            first = encoder(['meal'], [['meal.jpg']])
            second = encoder(['another meal'], [['other.jpg']])
        torch.testing.assert_close(first, second)
        self.assertEqual(call.call_args.args[1]['images'], ['other.jpg'])
        first.sum().backward()
        self.assertIsNotNone(encoder.projection[0].weight.grad)
        with patch('backends.action_encoder.HTTPBackend.call') as call:
            self.assertEqual(encoder(['']).abs().sum(), 0); call.assert_not_called()

    def test_pretrained_factor_generation(self):
        model = ActionVLM.__new__(ActionVLM); nn.Module.__init__(model)
        model.domain = 'exercise'
        model.vlm = SimpleNamespace(device='cpu', generate=lambda **kwargs:torch.tensor([[1,2,3,4]]))
        model.processor = SimpleNamespace(batch_decode=lambda *args, **kwargs:['```json\n{"factors":[20,70,3]}\n```'])
        with patch.object(model,'inputs',return_value=({'input_ids':torch.tensor([[1,2]])},None)) as inputs:
            factors = model.generate_factors('walking')
        torch.testing.assert_close(factors,torch.tensor([20.,70.,3.]))
        self.assertIn('intensity_MET',inputs.call_args.args[0])
        model.processor.batch_decode = lambda *args, **kwargs:['{"factors":[20,null,3]}']
        with patch.object(model,'inputs',return_value=({'input_ids':torch.tensor([[1,2]])},None)):
            with self.assertRaises(ValueError): model.generate_factors('walking')

    def test_reject_wrong_artifact_and_scale(self):
        encoder = HTTPActionEncoder('exercise', 8, 'other')
        with patch('backends.action_encoder.HTTPBackend.call', side_effect=self.response):
            with self.assertRaises(ValueError): encoder(['walking'])
        encoder.backend_id = 'sft'
        bad = self.response('', {'domain':'exercise'}); bad['result']['intensity_scale'] = 'Borg6_20'
        with patch('backends.action_encoder.HTTPBackend.call', return_value=bad):
            with self.assertRaises(ValueError): encoder(['walking'])

    def test_pooling_excludes_supervision_and_trains_backbone(self):
        class Backbone(nn.Module):
            def __init__(self): super().__init__(); self.weight = nn.Parameter(torch.ones(1))
            def forward(self, input_ids, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).repeat(1,1,4) * self.weight
                return SimpleNamespace(hidden_states=[hidden], loss=self.weight.square().mean())
        model = ActionVLM.__new__(ActionVLM); nn.Module.__init__(model)
        model.vlm = Backbone(); model.factor_head = nn.Linear(4, 3)
        first = model({'input_ids':torch.tensor([[1,2,100]])}, torch.tensor([[1,1,0]]))['factors']
        second = model({'input_ids':torch.tensor([[1,2,999]])}, torch.tensor([[1,1,0]]), torch.tensor([[2.,3.,float('nan')]]))
        torch.testing.assert_close(first, second['factors'])
        second['loss'].backward()
        self.assertIsNotNone(model.vlm.weight.grad); self.assertIsNotNone(model.factor_head.weight.grad)

    def test_image_prompt_and_annotation_mask(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                tokens = [1,2,3] if len(messages) == 1 else [1,2,3,4,5]
                return {'input_ids':torch.tensor([tokens]),'attention_mask':torch.ones(1,len(tokens),dtype=torch.long)}
        model = ActionVLM.__new__(ActionVLM); nn.Module.__init__(model)
        model.domain = 'diet'; model.processor = Processor()
        inputs, mask = model.inputs('', ['meal.jpg'], {'ingredients':['rice']})
        self.assertEqual(model.processor.messages[0]['content'][0]['type'], 'image')
        self.assertTrue(model.processor.messages[0]['content'][0]['path'].endswith('meal.jpg'))
        self.assertEqual(inputs['labels'].tolist(), [[-100,-100,-100,4,5]])
        self.assertEqual(mask.tolist(), [[1,1,1,0,0]])
        model.domain = 'exercise'
        with self.assertRaises(ValueError): model.inputs('walk', ['meal.jpg'])

    def test_world_and_preference_training(self):
        torch.set_num_threads(1)
        with patch.dict(os.environ, {'XLIFE_USE_LLM_ACTION_ENCODERS':'1'}), patch('backends.action_encoder.HTTPBackend.call', side_effect=self.response):
            model = MetabolicWorldModel(dimension=8,basis_dim=4,heads=2,layers=1,action_backend_id='sft')
            context = dict(history=[5.]*36,history_mask=[True]*36,metadata=encoded_metadata(),subject_id='s',origin='2025-01-01T12:00:00')
            action = dict(mode='diet',diet=[1.]*6,onset=[0,0],diet_images=['meal.jpg'])
            sample = candidate(context, action, {})
            batch = batch_candidates([sample], 'cpu')
            model.configure_phase('single')
            model(batch, 'diet', horizon=2)['predicted_cgm'].sum().backward()
            self.assertIsNotNone(model.diet_vlm.projection[0].weight.grad)
            model.zero_grad(set_to_none=True); model.configure_phase('fusion')
            self.assertFalse(model.diet_vlm.projection[0].weight.requires_grad)
            model.configure_phase('professional'); model.eval()
            reference = FrozenRankingReference(model)
            pairs = [dict(preferred=sample,rejected=sample,preferred_mode='diet',rejected_mode='diet',pair_weight=1.)]
            loss, _ = preference_objective(model,pairs,reference,2.,'cpu'); loss.backward()
            self.assertIsNotNone(model.ranking_head.score[0].weight.grad)
            self.assertIsNone(model.diet_vlm.projection[0].weight.grad)


if __name__ == '__main__': unittest.main()
