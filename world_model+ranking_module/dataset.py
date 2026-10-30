"""HTTP dataset adapter. Raw data paths refer to the server filesystem."""
from torch.utils.data import Dataset
from backends.http import HTTPBackend
from configuration import settings
from data_transport import pack, unpack


class XLifeCleanedDataset(Dataset):
    def __init__(self, path, max_subjects=None, max_samples=None, schema=None,
                 backend=None, processing_mode='event_origin', processing_options=None):
        response = (backend or HTTPBackend()).call('data/prepare', dict(
            path=str(path), max_subjects=max_subjects, max_samples=max_samples,
            schema=schema, data_config=settings('data'), processing_mode=processing_mode,
            processing_options=processing_options or {}))
        result = response['result']
        if result['data_config'] != settings('data'): raise ValueError('Server data configuration mismatch')
        self.samples = unpack(result['samples'])
        self.skipped = result['skipped']
        self.processing = result['processing']

    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


class SampleNumericDataset(XLifeCleanedDataset):
    pass


def split_samples(samples, strategy, seed, splits=None):
    result = HTTPBackend().call('data/split', dict(samples=[{k:s[k] for k in ('subject_id','origin','mode')} for s in samples], strategy=strategy,
        seed=seed, splits=splits, data_config=settings('data')))['result']
    return {k:[samples[i] for i in indices] for k,indices in result['indices'].items()}, result['splits']


def temporal_overlap(groups):
    return HTTPBackend().call('data/overlap', dict(groups={k:[{f:s[f] for f in ('subject_id','origin')} for s in rows] for k,rows in groups.items()},
        data_config=settings('data')))['result']['count']
