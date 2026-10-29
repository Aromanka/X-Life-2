from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from knowledge import KnowledgeGraph
from runtime_schema import load_schema, supervision_values, supervision_loss, retrieve
from world_model import MetabolicWorldModel


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((Path(__file__).resolve().parents[2]/'extra_code/schema.json').read_text(encoding='utf-8'))

    def test_masked_multimodal_loss(self):
        self.schema['supervision'] = [dict(name='arbitrary', path=['row','values'], mask=['row','observed'], shape=[2,2], readout='sequence', loss='mse', weight=2.)]
        row = supervision_values(self.schema, {'row':{'values':[[1.,None],[3.,4.]], 'observed':[True,False]}})
        prediction = torch.tensor([[[3.,100.],[100.,100.]]], requires_grad=True)
        model = SimpleNamespace(schema=self.schema, supervision_heads={'arbitrary':{'diet':lambda x:prediction}})
        loss = supervision_loss(model, [row], {'trajectory':torch.zeros(1,2,2)}, 'diet')
        self.assertEqual(loss.item(), 8.)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[[8.,0.],[0.,0.]]]))
        absent = supervision_values(self.schema, {})
        with self.assertRaisesRegex(ValueError, 'No observed'): supervision_loss(model, [absent], {'trajectory':torch.zeros(1,2,2)}, 'diet')

    def test_auxiliary_heads_checkpoint_and_phases(self):
        self.schema['supervision'] = [dict(name='custom',path=['patient','new_source'],shape=[3],readout='pooled',loss='l1',weight=.5)]
        model = MetabolicWorldModel(dimension=8,basis_dim=4,heads=2,layers=1,schema=self.schema)
        model.configure_phase('fusion')
        self.assertTrue(model.supervision_heads['custom']['joint'].weight.requires_grad)
        self.assertFalse(model.supervision_heads['custom']['diet'].weight.requires_grad)
        trajectory = torch.randn(2,4,16,requires_grad=True)
        rows = [supervision_values(self.schema, {'patient':{'new_source':[1,2,3]}})]*2
        supervision_loss(model, rows, {'trajectory':trajectory}, 'joint').backward()
        self.assertIsNotNone(model.supervision_heads['custom']['joint'].weight.grad)
        self.assertIsNotNone(trajectory.grad)
        model.configure_phase('professional')
        self.assertFalse(model.supervision_heads['custom']['joint'].weight.requires_grad)
        restored = MetabolicWorldModel(dimension=8,basis_dim=4,heads=2,layers=1,schema=self.schema)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.supervision_heads['custom']['joint'].weight,model.supervision_heads['custom']['joint'].weight)

    def test_semantic_supervision(self):
        self.schema['supervision'] = [dict(name='semantic',path=['patient','annotations'],shape=[2],readout='pooled',loss='mse',weight=1.,encode={'role':'metadata/encode','input':'fields','output':'embedding'})]
        envelope = {'result':{'embedding':[1.,2.]},'provenance':{'test_backend':True}}
        with patch('runtime_schema.HTTPBackend.call',return_value=envelope) as call:
            row = supervision_values(self.schema, {'patient':{'annotations':[{'name':'x','value':'finding','unit':''}]}})
        self.assertTrue(row['semantic']['test_backend'])
        self.assertEqual(call.call_args.args[0], 'metadata/encode')

    def test_personalized_vector_query(self):
        graph = KnowledgeGraph([dict(head='a',relation='r',tail='b',source='test')])
        backend = SimpleNamespace(call=lambda role,payload:{'result':{'vector':[1.,0.]},'provenance':{'test_backend':True}})
        spec = dict(strategy='vector',paths=[['context'],['requirement']],dimension=2,limit=1)
        with patch.object(graph,'retrieve_vector',wraps=graph.retrieve_vector) as call:
            rows, provenance = retrieve(graph,spec,{'context':{'history':[5.], 'metadata':{'finding':'x'}},'requirement':{'preference':'p'}},backend)
        query = json.loads(call.call_args.args[0])
        self.assertEqual(query[0]['history'], [5.])
        self.assertEqual(query[1]['preference'], 'p')
        self.assertEqual(len(rows),1)
        self.assertTrue(provenance[0]['test_backend'])
        spec.update(strategy='exact',paths=[['anything']])
        self.assertEqual(retrieve(graph,spec,{'anything':['a']},backend)[0],graph.triples)

    def test_schema_validation(self):
        backend = SimpleNamespace(call=lambda role,payload:{'result':self.schema})
        self.assertEqual(load_schema(backend), self.schema)
        invalid = deepcopy(self.schema); invalid['supervision'][0]['weight'] = -1
        backend.call = lambda role,payload:{'result':invalid}
        with self.assertRaises(ValueError): load_schema(backend)


if __name__ == '__main__': unittest.main()
