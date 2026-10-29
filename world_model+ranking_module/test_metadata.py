from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from backends.metadata_encoder import SemanticMetadataEncoder
from dataset_metadata import metadata_fields, prepare_metadata
from world_model import MetabolicWorldModel


def encoded_metadata(width=5):
    return dict(fields=[dict(name='test',value=1.,unit='')],embedding=[.1]*width,backend_id='fixture-qwen3',test_backend=True)


class MetadataTests(unittest.TestCase):
    def test_arbitrary_sources_and_widths(self):
        source = {'anthropometry':[170.,70.], 'lab':{'result':'negative'}, 'sensor':[1.,None,3.], 'omics':{'protein':{'value':2.,'unit':'ng/mL'}}}
        fields = metadata_fields(source)['fields']
        self.assertEqual(len(fields), 7)
        self.assertEqual(next(f for f in fields if f['name']=='omics.protein')['unit'], 'ng/mL')
        self.assertEqual(next(f for f in fields if f['name']=='lab.result')['value'], 'negative')
        self.assertIsNone(next(f for f in fields if f['name']=='sensor[1]')['value'])
        self.assertEqual(metadata_fields({'new_cohort':[1,2,3,4]})['fields'][-1]['name'], 'new_cohort[3]')
        self.assertEqual(metadata_fields({'fields':fields})['fields'], fields)

    def test_remote_identity_and_validation(self):
        result = {'result':{'embedding':[1.,2.,3.], 'backend_id':'qwen3'},'provenance':{'test_backend':False}}
        with patch('dataset_metadata.HTTPBackend.call',return_value=result) as call:
            encoded = prepare_metadata({'clinical':{'HbA1c':{'value':6.1,'unit':'%'}}})
        self.assertEqual(call.call_args.args[0], 'metadata/encode')
        self.assertEqual(encoded['embedding'], [1.,2.,3.])
        with patch('dataset_metadata.HTTPBackend.call') as call:
            self.assertIs(prepare_metadata(encoded), encoded); call.assert_not_called()
        invalid = deepcopy(encoded); invalid['embedding'] = [float('nan')]
        with self.assertRaises(ValueError): prepare_metadata(invalid)

    def test_alignment_and_checkpoint_infer_dimension(self):
        encoder = SemanticMetadataEncoder(4)
        loss = encoder([encoded_metadata(7)]).sum(); loss.backward()
        self.assertEqual(encoder.projection.weight.shape, (4,7))
        self.assertIsNotNone(encoder.projection.weight.grad)
        restored = SemanticMetadataEncoder(4); restored.load_state_dict(encoder.state_dict())
        torch.testing.assert_close(restored([encoded_metadata(7)]), encoder([encoded_metadata(7)]))
        wrong = encoded_metadata(7); wrong['backend_id'] = 'other'
        with self.assertRaises(ValueError): restored([wrong])
        with self.assertRaises(RuntimeError): restored([encoded_metadata(9)])
        with self.assertRaises(ValueError): MetabolicWorldModel().load_checkpoint({'model_state_dict':{'metadata_encoder.mean':torch.zeros(1)}})


if __name__ == '__main__': unittest.main()
