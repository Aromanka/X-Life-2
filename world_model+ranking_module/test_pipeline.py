"""Contract tests use a real local POST server; no upstream APIs or weights."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import torch
from backends.http import HTTPBackend,BackendError
from experiments.baselines import persistence,linear_history,DLinear
from experiments.tasks import task_window
from generation import validate_candidate
from inference import recommend,masked_context
from knowledge import KnowledgeGraph
from world_model import MetabolicWorldModel


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        root=Path(__file__).resolve().parents[2]
        cls.server=subprocess.Popen([sys.executable,str(root/'extra_code/server.py'),'--backend','mock','--port',str(port)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        cls.backend=HTTPBackend(f'http://127.0.0.1:{port}',timeout=3)
        for _ in range(100):
            try:
                cls.backend.call('teacher',{});break
            except BackendError:
                if cls.server.poll() is not None: raise RuntimeError('Mock server exited')
                time.sleep(.05)
        else: raise RuntimeError('Mock server failed to start')

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate();cls.server.wait(timeout=10)

    def setUp(self):
        torch.manual_seed(42)
        self.model=MetabolicWorldModel(dimension=16,basis_dim=8,heads=2,layers=1).eval()
        self.model.initialize_fusion()
        self.checkpoint={'phase':'fusion','vocabulary':{'walking':1},'trained_domains':['diet','exercise']}
        self.context=dict(subject_id='fixture',origin='2025-01-01T12:00:00',history=[5.]*36,history_mask=[True]*36,metadata=[None]*84)

    def run_pipeline(self,**kwargs):
        return recommend(self.context,self.model,self.checkpoint,self.backend,KnowledgeGraph(),ranking='random',**kwargs)

    def test_http_roles(self):
        for role,payload in [('action/latent',{'dimension':16,'text':'test'}),('embed',{'dimension':16,'text':'test'}),('knowledge/extract',{'text':'test'})]:
            response=self.backend.call(role,payload)
            self.assertTrue(response['provenance']['test_backend'])
        with self.assertRaises(BackendError): self.backend.call('nonexistent',{})

    def test_three_modes_and_immutable_context(self):
        before=deepcopy(self.context)
        for mode in ('diet','exercise','joint'):
            result=self.run_pipeline(mode=mode)
            self.assertEqual(len(result['top_plans']),3)
            self.assertTrue(result['test_only'])
            self.assertEqual(result['context'],before)
            self.assertEqual(result['top_plans'][0]['simulation']['time_minutes'][-1],180)
            self.assertEqual(len(result['top_plans'][0]['simulation']['predicted_cgm']),36)

    def test_hard_veto_overrides_semantic_acceptance(self):
        result=self.run_pipeline(mode='joint',constraints={'allergies':['rice'],'min_exercise_delay_min':30})
        self.assertEqual(result['top_plans'],[])
        for row in result['candidates']:
            self.assertIn('allergy_veto',row['assessment']['hard_vetoes'])
            self.assertIn('joint_exercise_timing',row['assessment']['hard_vetoes'])

    def test_migrated_policy_and_knowledge(self):
        raw=self.backend.call('generate',{'mode':'joint','count':1})['result']['candidates']
        raw[0]['exercise_type']='sprint'
        graph=KnowledgeGraph([{'head':'sprint','relation':'avoid_for','tail':'heart_disease','source':'fixture'}])
        result=recommend(self.context,self.model,self.checkpoint,self.backend,graph,ranking='random',candidates=raw,
                         constraints={'conditions':['heart_disease']})
        self.assertEqual(result['top_plans'],[])
        self.assertTrue(result['candidates'][0]['safety_knowledge'])
        self.assertTrue(result['candidates'][0]['assessment']['hard_vetoes'])

    def test_small_count_and_failure(self):
        result=self.run_pipeline(count=2)
        self.assertEqual(len(result['top_plans']),2)
        with patch.object(self.backend,'call',side_effect=BackendError('offline')):
            result=self.run_pipeline()
        self.assertEqual(result['top_plans'],[])
        self.assertEqual(result['failures'][0]['stage'],'generation')

    def test_malformed_and_review_outage_fail_closed(self):
        raw=self.backend.call('generate',{'mode':'diet','count':1})['result']['candidates']
        broken=deepcopy(raw[0]);broken['units']={}
        self.assertEqual(self.run_pipeline(candidates=[broken])['candidates'][0]['status'],'invalid')
        with patch.object(self.backend,'call',side_effect=BackendError('offline')):
            result=self.run_pipeline(candidates=raw)
        self.assertEqual(result['top_plans'],[])
        self.assertEqual(result['candidates'][0]['status'],'rejected')
        with patch.object(self.backend,'call',return_value={'result':{'is_safe':'yes','reasons':[]},'provenance':{'test_backend':True}}):
            self.assertEqual(self.run_pipeline(candidates=raw)['top_plans'],[])

    def test_learned_ranking_never_falls_back(self):
        result=recommend(self.context,self.model,self.checkpoint,self.backend,KnowledgeGraph(),ranking='learned')
        self.assertEqual(result['top_plans'],[])
        self.assertIn('No trained ranking',result['candidates'][0]['reason'])
        self.model.ranked_modes.fill_(True)
        result=recommend(self.context,self.model,self.checkpoint,self.backend,KnowledgeGraph(),ranking='learned')
        self.assertEqual(len(result['top_plans']),3)
        self.assertEqual(result['ranking_backend'],'learned')

    def test_mask_allowlist(self):
        sample={**self.context,'diet':[99]*6,'exercise':[88]*3,'target':[999]*36,'diet_text':'secret','onset':[100,0]}
        self.assertEqual(set(masked_context(sample)),set(self.context))
        self.assertNotIn('secret',json.dumps(masked_context(sample)))
        with self.assertRaises(ValueError):
            recommend(sample,self.model,self.checkpoint,self.backend,KnowledgeGraph())

    def test_http_action_encoder(self):
        with patch.dict(os.environ,{'XLIFE_LLM_URL':self.backend.url,'XLIFE_USE_LLM_ACTION_ENCODERS':'1'}):
            from backends.action_encoder import HTTPActionEncoder
            encoder=HTTPActionEncoder('diet',16,32,'random-latent-v1')
            a=encoder(['meal']);b=encoder(['meal'])
            torch.testing.assert_close(a,b)
            a.sum().backward();self.assertIsNotNone(encoder.projection.weight.grad)
            self.assertTrue(encoder.test_backend)
            encoder.backend_id='wrong'
            with self.assertRaises(ValueError): encoder(['meal'])

    def test_baseline_math_and_training(self):
        values=torch.arange(36,dtype=torch.float32)[None];mask=torch.ones_like(values,dtype=torch.bool)
        torch.testing.assert_close(persistence(values,mask),torch.full((1,36),35.))
        torch.testing.assert_close(linear_history(values,mask),torch.arange(36,72,dtype=torch.float32)[None])
        mask[:,10:20]=False
        model=DLinear();batch={'history':values,'history_mask':mask}
        model(batch).square().mean().backward()
        self.assertIsNotNone(model.trend.weight.grad)

    def test_controlled_architecture_variants(self):
        from alignment import candidate,batch_candidates
        action=dict(mode='diet',diet=[1.]*6,exercise=[None]*3,onset=[0,0])
        batch=batch_candidates([candidate(self.context,action,{})],'cpu')
        for config in ({'patch_size':3},{'state_pooling':'recent'},{'gap_feature':False},{'readout':'autoregressive'}):
            model=MetabolicWorldModel(dimension=16,basis_dim=8,heads=2,layers=1,**config)
            prediction=model(batch,'diet',horizon=3)['predicted_cgm']
            self.assertEqual(prediction.shape,(1,3));self.assertTrue(torch.isfinite(prediction).all())

    def test_task_definition_required(self):
        record=dict(task='night',origin='2025-01-01T23:00:00',subject_id='fixture',history_minutes=10,target_minutes=10,
            history_points=[['2025-01-01T22:55:00',5]],target_points=[['2025-01-01T23:05:00',6]],glucose_unit='mmol/L',label_source='fixture')
        with self.assertRaises(ValueError): task_window(record)
        record.update(bedtime=record['origin'],wake_time='2025-01-01T23:10:00')
        self.assertEqual(int(task_window(record)['target_mask'].sum()),1)


if __name__=='__main__':unittest.main()
