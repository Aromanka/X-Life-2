import os
import unittest
from datetime import datetime
from unittest.mock import patch
import torch
from dataset import cgm_window
from world_model import MetabolicWorldModel, llm_actions_enabled
from train import evaluate


class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        self.model = MetabolicWorldModel(dimension=16, basis_dim=8, heads=2, layers=1)
        self.batch = dict(history=torch.arange(36).float()[None]/10+5, history_mask=torch.ones(1,36,dtype=torch.bool), metadata=torch.zeros(1,84), diet=torch.ones(1,6), exercise=torch.ones(1,3), type_id=torch.zeros(1,dtype=torch.long), onset=torch.tensor([[60.,0.]]))

    def test_time_grid(self):
        origin = datetime.fromisoformat('2025-01-01T12:00:00')
        points = [['2025-01-01T11:55:00',5.], ['2025-01-01T12:00:00',6.], ['2025-01-01T12:05:00',7.], ['2025-01-01T12:15:00',8.]]
        past, pm = cgm_window(points, origin)
        future, fm = cgm_window(points, origin, True)
        self.assertEqual(pm.sum(),1); self.assertEqual(past[-1],5.)
        self.assertEqual(future[0],7.); self.assertFalse(fm[1]); self.assertEqual(future[2],8.)

    def test_joint_update(self):
        self.model.initialize_fusion()
        self.assertTrue(torch.equal(self.model.fusion.basis, torch.block_diag(self.model.diet.basis,self.model.exercise.basis)))
        d,e,_ = self.model.initial_states(self.batch)
        state = torch.cat([d,e],-1)
        gate = torch.sigmoid(self.model.gate(state @ self.model.fusion.basis.T))
        expected = state+torch.cat([gate[:,:1]*self.model.diet.increment(d),gate[:,1:2]*self.model.exercise.increment(e)],-1)+gate[:,2:]*self.model.fusion.increment(state)
        output = self.model(self.batch,'joint',horizon=1)
        torch.testing.assert_close(output['states'][:,0],expected)
        self.assertEqual(output['gates'].shape,(1,1,3))

    def test_freeze_and_reload(self):
        self.model.initialize_fusion(); self.model.configure_phase('fusion')
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        optim = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=.001)
        self.model(self.batch,'joint',horizon=3)['predicted_cgm'].square().mean().backward(); optim.step()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: torch.testing.assert_close(parameter,before[name],rtol=0,atol=0)
        self.assertFalse(torch.equal(self.model.fusion.decoder.weight,before['fusion.decoder.weight']))
        restored = MetabolicWorldModel(dimension=16,basis_dim=8,heads=2,layers=1)
        restored.load_state_dict(self.model.state_dict())
        for mode in ('diet','exercise','joint'):
            prediction = self.model.simulate(self.batch,mode)['predicted_cgm']
            self.assertEqual(prediction.shape,(1,36))
            torch.testing.assert_close(prediction,restored.simulate(self.batch,mode)['predicted_cgm'])

    def test_no_target_input(self):
        first = self.model.simulate(self.batch,'diet')['predicted_cgm']
        self.batch['target'] = torch.full((1,36),1000.)
        torch.testing.assert_close(first,self.model.simulate(self.batch,'diet')['predicted_cgm'])

    def test_backend_flag(self):
        with patch.dict(os.environ,{'XLIFE_USE_LLM_ACTION_ENCODERS':'0'}): self.assertFalse(llm_actions_enabled())
        with patch.dict(os.environ,{'XLIFE_USE_LLM_ACTION_ENCODERS':'1'}):
            with self.assertRaises(ValueError): MetabolicWorldModel(dimension=16,heads=2)
        with patch.dict(os.environ,{'XLIFE_USE_LLM_ACTION_ENCODERS':'invalid'}):
            with self.assertRaises(ValueError): llm_actions_enabled()

    def test_metrics(self):
        sample = {key:value[0] for key,value in self.batch.items()}
        sample.update(subject_id='test', origin='2025-01-01T12:00:00', mode='diet', joint=False, exercise_type='', diet_text='', exercise_text='', target=torch.tensor([3.,5.,7.]+[0.]*33), target_mask=torch.tensor([True]*3+[False]*33))
        with patch.object(self.model,'forward',return_value={'predicted_cgm':torch.tensor([[1.,2.,3.]+[0.]*33])}):
            metrics,_ = evaluate(self.model,[sample],{},'cpu',1,'diet')
        self.assertAlmostEqual(metrics['mae'],3.)
        self.assertAlmostEqual(metrics['pearson'],1.)
        self.assertEqual(metrics['valid_points'],3)


if __name__ == '__main__': unittest.main()
