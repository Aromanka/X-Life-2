import os
from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import torch
from test_metadata import encoded_metadata
from dataset import cgm_window
from world_model import MetabolicWorldModel, llm_actions_enabled
from train import evaluate, split_samples, temporal_overlap, curve_metrics, main
from runtime_schema import supervision_values


class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        self.model = MetabolicWorldModel(dimension=16, basis_dim=8, heads=2, layers=1)
        self.batch = dict(history=torch.arange(36).float()[None]/10+5, history_mask=torch.ones(1,36,dtype=torch.bool), metadata=[encoded_metadata()], diet=torch.ones(1,6), exercise=torch.ones(1,3), type_id=torch.zeros(1,dtype=torch.long), onset=torch.tensor([[60.,0.]]))
        self.model.initial_states(self.batch)

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
        before = deepcopy(self.model.state_dict())
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
        self.assertEqual(metrics['domains']['diet']['mae'],3.)
        self.assertEqual(metrics['horizons']['30']['mae'],3.)

    def test_temporal_split(self):
        rows = [dict(subject_id=s,origin=f'2025-01-01T{h:02}:00:00',mode='diet') for s in ('a','b') for h in (14,8,12,10)]
        groups,splits = split_samples(rows,'within_subject',42)
        self.assertEqual(len(groups['train']),4); self.assertEqual(len(groups['validation']),4)
        self.assertEqual(groups['test'],[])
        for subject in ('a','b'):
            self.assertLess(max(s['origin'] for s in groups['train'] if s['subject_id']==subject),min(s['origin'] for s in groups['validation'] if s['subject_id']==subject))
        restored,_ = split_samples(list(reversed(rows)),'within_subject',42,splits)
        self.assertEqual({s['origin']+s['subject_id'] for s in restored['train']},{s['origin']+s['subject_id'] for s in groups['train']})
        self.assertGreater(temporal_overlap(groups),0)

    def test_subject_split(self):
        rows = [dict(subject_id=str(i),origin='2025-01-01T12:00:00',mode='diet') for i in range(20)]
        groups,_ = split_samples(rows,'subject',42)
        self.assertEqual([len(groups[k]) for k in ('train','validation','test')],[16,2,2])
        self.assertFalse({s['subject_id'] for s in groups['train']} & {s['subject_id'] for s in groups['validation']})

    def test_missing_horizon(self):
        row=dict(predicted=[2.]*36,observed=[1.]*36,mask=[False]*6+[True]*30,last_observed=1.)
        metrics=curve_metrics([row])
        self.assertIsNone(metrics['horizons']['30']['mae'])
        self.assertEqual(metrics['persistence_mae'],0.)
        self.assertEqual(metrics['invalid_correlations'],1)

    def test_selection_and_training_control(self):
        schema = json.loads((Path(__file__).resolve().parents[2]/'extra_code/schema.json').read_text(encoding='utf-8'))
        class Rows:
            skipped = 0
            def __len__(self): return len(self.samples)
        dataset = Rows(); dataset.samples = []
        for i,hour in enumerate((8,10,12,14)):
            row = {key:deepcopy(value[0]) for key,value in self.batch.items()}
            row.update(subject_id='a',origin=f'2025-01-01T{hour:02}:00:00',mode='diet' if i%2==0 else 'exercise',joint=True,exercise_type='',diet_text='',exercise_text='',target=torch.ones(36)*6,target_mask=torch.ones(36,dtype=torch.bool))
            row['supervision'] = supervision_values(schema, {'sample':row})
            dataset.samples.append(row)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'config.json'
            config.write_text(json.dumps({'model':{'dimension':16,'basis_dim':8,'heads':2,'layers':1}}))
            args = ['--data','unused','--config',str(config),'--split-strategy','within_subject','--epochs','20','--batch-size','1','--device','cpu','--output',directory]
            with patch('train.load_schema',return_value=schema),patch('train.XLifeCleanedDataset',return_value=dataset),patch('train.evaluate',return_value=({'mae':1.},[])),patch('builtins.print'):
                main(args)
            best = torch.load(Path(directory)/'best.pt',weights_only=False)
            last = torch.load(Path(directory)/'checkpoint.pt',weights_only=False)
            self.assertEqual(best['epoch'],1); self.assertEqual(last['epoch'],11)
            self.assertEqual(last['bad_epochs'],10)
            self.assertLess(last['optimizer']['param_groups'][0]['lr'],.0001)
            with patch('train.load_schema',return_value=schema),patch('train.XLifeCleanedDataset',return_value=dataset),patch('train.evaluate',return_value=({'mae':.8},[])),patch('builtins.print'):
                main(args+['--resume',str(Path(directory)/'checkpoint.pt'),'--epochs','1'])
            resumed = torch.load(Path(directory)/'checkpoint.pt',weights_only=False)
            self.assertEqual(resumed['epoch'],12); self.assertEqual(resumed['bad_epochs'],0)
            self.assertEqual(resumed['best_mae'],.8)


if __name__ == '__main__': unittest.main()
