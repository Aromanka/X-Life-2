import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_metadata import encoded_metadata
from alignment import NumericPreferencePairs, FrozenRankingReference, preference_objective, batch_candidates
from train import main
from world_model import MetabolicWorldModel, RankingHead


class AlignmentTests(unittest.TestCase):
    def test_standard_dpo_matches_full_candidate_policy(self):
        logits = torch.tensor([[2.,-1.,5.],[-3.,4.,1.]],requires_grad=True)
        reference = torch.tensor([[1.,0.,-2.],[2.,-1.,3.]])
        pairs = [dict(pair_weight=1.),dict(pair_weight=2.)]
        def scores(model,pairs,side,ref,device):
            index = 0 if side=='preferred' else 1
            return logits[:,index],reference[:,index]
        with patch('alignment.pair_scores',side_effect=scores): loss,_ = preference_objective(None,pairs,None,2.,'cpu')
        ratio = logits.log_softmax(-1)-reference.log_softmax(-1)
        expected = (-F.logsigmoid(2*(ratio[:,0]-ratio[:,1]))*torch.tensor([1.,2.])).sum()/3
        torch.testing.assert_close(loss,expected)
        torch.testing.assert_close(torch.autograd.grad(loss,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0])
        self.assertFalse(reference.requires_grad)

    def test_unbounded_logits_and_wrong_order_gradient(self):
        head = RankingHead(8,2,1).eval()
        with torch.no_grad(): head.score[-1].weight.zero_(); head.score[-1].bias.fill_(3.)
        self.assertTrue((head(torch.zeros(2,3,8),'diet')>1).all())
        with torch.no_grad(): head.score[-1].bias.fill_(-3.)
        self.assertTrue((head(torch.zeros(2,3,8),'diet')<0).all())
        logits = torch.tensor([-20.,20.],requires_grad=True)
        def scores(model,pairs,side,ref,device): return logits[0 if side=='preferred' else 1].reshape(1),torch.zeros(1)
        with patch('alignment.pair_scores',side_effect=scores): loss,_ = preference_objective(None,[dict(pair_weight=1.)],None,2.,'cpu')
        loss.backward()
        self.assertLess(float(logits.grad[0]),-1.9)
        with patch('alignment.pair_scores',side_effect=lambda m,p,s,r,d:(torch.tensor([20. if s=='preferred' else -20.]),torch.zeros(1))):
            loss,_ = preference_objective(None,[dict(pair_weight=1.)],None,2.,'cpu')
        self.assertLess(float(loss),1e-10)

    def test_bounded_alignment_checkpoint_rejected(self):
        checkpoint = dict(ranking_version=1,phase='professional',model_state_dict=self.model.state_dict())
        with self.assertRaisesRegex(ValueError,'restart alignment'): self.model.load_checkpoint(checkpoint)
        checkpoint['phase'] = 'fusion'
        self.model.load_checkpoint(checkpoint)

    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(3)
        self.config = dict(dimension=16, basis_dim=8, heads=2, layers=1)
        self.model = MetabolicWorldModel(**self.config)
        self.context = dict(history=[5.+i/100 for i in range(36)], history_mask=[True]*36, metadata=encoded_metadata(), subject_id='test', origin='2025-01-01T12:00:00')
        self.action = dict(mode='diet', diet=[1.]*6, exercise=[2.]*3, onset=[0.,0.])
        self.model.metadata_encoder([self.context['metadata']])
        self.row = dict(state_id='s1', split='train', context=self.context, preferred=self.action, rejected={**self.action, 'diet':[10.]*6})

    def test_score_direction_reference_and_frozen_predictions(self):
        pairs = NumericPreferencePairs([self.row], {}, 'professional')
        self.model.configure_phase('professional'); self.model.eval()
        reference = FrozenRankingReference(self.model)
        batch = batch_candidates([pairs[0]['preferred']], 'cpu')
        before = deepcopy(self.model.state_dict())
        predictions = {m:self.model.simulate(batch,m)['predicted_cgm'] for m in ('diet','exercise','joint')}
        trajectory = self.model(batch,'diet')['trajectory']
        ref_score = reference(trajectory,'diet').clone()
        initial, _ = preference_objective(self.model, list(pairs), reference, 2., 'cpu')
        self.assertAlmostEqual(float(initial.detach()), .693147, places=5)
        optimizer = torch.optim.AdamW(self.model.ranking_head.parameters(), lr=.005)
        for _ in range(15):
            optimizer.zero_grad(); loss,_ = preference_objective(self.model,list(pairs),reference,2.,'cpu'); loss.backward(); optimizer.step()
        final, accuracy = preference_objective(self.model,list(pairs),reference,2.,'cpu')
        self.assertLess(float(final.detach()), float(initial.detach())); self.assertEqual(float(accuracy),1.)
        torch.testing.assert_close(reference(trajectory,'diet'),ref_score,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        for key,value in self.model.state_dict().items():
            if key.endswith('_extra_state'): self.assertEqual(value,before[key])
            elif not key.startswith('ranking_head.'): torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        for mode in predictions:
            result = self.model.simulate(batch,mode)
            torch.testing.assert_close(result['predicted_cgm'],predictions[mode],rtol=0,atol=0)
            self.assertTrue(torch.isfinite(result['ranking_logit']).all())

    def test_data_guards(self):
        invalid = deepcopy(self.row); invalid['rejected']['target'] = [6.]*36
        with self.assertRaises(ValueError): NumericPreferencePairs([invalid],{},'professional')
        invalid = deepcopy(self.row); invalid['preferred']['metadata'] = {'unexpected':'context'}
        with self.assertRaises(ValueError): NumericPreferencePairs([invalid],{},'professional')
        with self.assertRaises(ValueError): NumericPreferencePairs([self.row],{},'physiological')
        invalid = deepcopy(self.row); invalid.update(split='validation',state_id='alias')
        with self.assertRaises(ValueError): NumericPreferencePairs([self.row,invalid],{},'professional')
        pair = NumericPreferencePairs([self.row],{},'professional')[0]
        self.assertNotIn('target',pair['preferred'])
        torch.testing.assert_close(pair['preferred']['history'],pair['rejected']['history'])

    def test_legacy_checkpoint_and_rank_guard(self):
        state = {k:v for k,v in self.model.state_dict().items() if not k.startswith('ranking_head.') and k!='ranked_modes'}
        restored = MetabolicWorldModel(**self.config); restored.load_checkpoint(dict(model_state_dict=state))
        pairs = NumericPreferencePairs([self.row],{},'professional')
        batch = batch_candidates([pairs[0]['preferred']],'cpu')
        for mode in ('diet','exercise','joint'):
            torch.testing.assert_close(self.model.simulate(batch,mode)['predicted_cgm'],restored.simulate(batch,mode)['predicted_cgm'],rtol=0,atol=0)
        with self.assertRaises(ValueError): restored.rank(batch,['diet'])
        del state['gate.bias']
        with self.assertRaises(ValueError): restored.load_checkpoint(dict(model_state_dict=state))

    def test_stage_chain_replay_resume_and_mixed_ranking(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            root = Path(directory)
            parent = root/'fusion.pt'
            torch.save(dict(architecture_version=3, ranking_version=2, model_config=self.config, model_state_dict=self.model.state_dict(), llm_actions=False, phase='fusion', vocabulary={}, config={}, trained_domains=['diet','exercise']),parent)
            rows = []
            for i,mode in enumerate(('diet','exercise','joint')):
                row = deepcopy(self.row); row['state_id'] = str(i); row['context']['origin'] = f'2025-01-0{i+1}T12:00:00'
                row['preferred']['mode'] = mode; row['rejected']['mode'] = mode
                row['rejected']['exercise'] = [8.]*3; rows.append(row)
            expert = root/'expert.json'; expert.write_text(json.dumps(rows))
            physiological = root/'physiological.json'
            physical = deepcopy(rows)
            for row in physical: row['evidence'] = dict(source='synthetic test only',preferred_observation_id='a',rejected_observation_id='b')
            physiological.write_text(json.dumps(physical))
            base = ['--device','cpu','--batch-size','2','--epochs','1']
            pro = root/'professional'; phy = root/'physiological'
            with self.assertRaises(ValueError): main(base+['--phase','physiological','--data',str(physiological),'--init-checkpoint',str(parent),'--output',str(phy)])
            main(base+['--phase','professional','--data',str(expert),'--init-checkpoint',str(parent),'--output',str(pro)])
            expert_state = torch.load(pro/'checkpoint.pt',weights_only=False)
            args = base+['--phase','physiological','--data',str(physiological),'--replay-data',str(expert),'--output',str(phy)]
            main(args+['--init-checkpoint',str(pro/'checkpoint.pt')])
            first = torch.load(phy/'checkpoint.pt',weights_only=False)
            self.assertEqual(first['history'][0]['replay_fraction'],.25)
            for key,value in first['reference_state_dict'].items(): torch.testing.assert_close(value,expert_state['model_state_dict'][key],rtol=0,atol=0)
            main(args+['--resume',str(phy/'checkpoint.pt')])
            resumed = torch.load(phy/'checkpoint.pt',weights_only=False)
            self.assertEqual(resumed['epoch'],2)
            for key,value in first['reference_state_dict'].items(): torch.testing.assert_close(value,resumed['reference_state_dict'][key],rtol=0,atol=0)
            for key,value in resumed['model_state_dict'].items():
                if key.endswith('_extra_state'): self.assertEqual(value,self.model.state_dict()[key])
                elif not key.startswith('ranking_head.') and key != 'ranked_modes': torch.testing.assert_close(value,self.model.state_dict()[key],rtol=0,atol=0)
            candidates = root/'candidates.json'; candidates.write_text(json.dumps([dict(state_id='rank',context=self.context,candidates=[r['preferred'] for r in rows])]))
            main(base+['--rank-only','--data',str(candidates),'--init-checkpoint',str(phy/'checkpoint.pt'),'--output',str(root/'rank')])
            result = json.loads((root/'rank/rankings.json').read_text())[0]
            restored = MetabolicWorldModel(**self.config); restored.load_checkpoint(resumed)
            samples = [NumericPreferencePairs([r],{},'professional')[0]['preferred'] for r in rows]
            expected = [float(restored.simulate(batch_candidates([s],'cpu'),mode)['ranking_logit'][0]) for s,mode in zip(samples,('diet','exercise','joint'))]
            self.assertNotIn('scores', result)
            self.assertEqual(result['order'], sorted(range(len(expected)),key=lambda i:expected[i],reverse=True))


if __name__ == '__main__': unittest.main()
