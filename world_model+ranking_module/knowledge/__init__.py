"""Local graph snapshots with separate generation and safety retrieval strategies."""
import json
import re
import unicodedata
from pathlib import Path
from configuration import configured


def normalize(value):
    return unicodedata.normalize('NFKC', str(value)).strip().casefold()


class KnowledgeGraph:
    def __init__(self, triples=(), source='explicit-empty', provenance=()):
        self.source = source
        self.provenance=list(provenance)
        self._vector_index = None
        self.triples = []
        for row in triples:
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ('head','relation','tail','source')):
                raise ValueError('Knowledge triples require head, relation, tail and evidence source')
            self.triples.append(dict(row))
        self.index = {}
        for i,row in enumerate(self.triples):
            for term in (row['head'],row['tail']): self.index.setdefault(normalize(term),set()).add(i)

    @classmethod
    def load(cls, path):
        if not path: return cls()
        path=Path(path);sidecar=path.with_suffix('.provenance.json')
        provenance=json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else []
        return cls(json.loads(path.read_text(encoding='utf-8')),str(path),provenance)

    def retrieve(self, terms, strategy=None, limit=None):
        strategy = configured('knowledge', 'strategy', strategy)
        limit = configured('knowledge', 'limit', limit)
        terms = [normalize(t) for t in terms if t]
        if strategy == 'safety':
            indices = set().union(*(self.index.get(t,set()) for t in terms)) if terms else set()
        elif strategy == 'generation':
            indices = {i for i,row in enumerate(self.triples) if any(t in normalize(row['head']+' '+row['tail']) for t in terms)}
        else: raise ValueError('Unknown retrieval strategy')
        return [self.triples[i] for i in sorted(indices)[:limit]]

    def retrieve_vector(self, query, backend, dimension=None, limit=None):
        dimension = configured('knowledge', 'dimension', dimension)
        limit = configured('knowledge', 'limit', limit)
        if not self.triples: return [], []
        import numpy as np
        entities = dict.fromkeys(normalize(row[key]) for row in self.triples for key in ('head', 'tail'))
        entities = list(entities)
        def identity(response):
            return (response['result']['backend_id'], response['result'].get('model_revision'),
                    response['provenance']['configuration_id'], response['provenance']['test_backend'])
        def normalized(responses):
            vectors = np.asarray([r['result']['vector'] for r in responses], dtype=float)
            if vectors.shape != (len(responses),dimension) or not np.isfinite(vectors).all(): raise ValueError('Invalid embeddings')
            norms = np.linalg.norm(vectors,axis=1,keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any(): raise ValueError('Knowledge embeddings require finite nonzero norms')
            return vectors/norms
        response = backend.call('embed', {'text':query, 'dimension':dimension, 'input_type':'query'})
        query_vector = normalized([response])[0]
        index_key = (identity(response), dimension, tuple(entities))
        if self._vector_index is None or self._vector_index[0] != index_key:
            documents = [backend.call('embed', {'text':text, 'dimension':dimension, 'input_type':'document'}) for text in entities]
            if any(identity(row) != identity(response) for row in documents):
                raise ValueError('Knowledge query and documents require the same embedding model and configuration')
            entity_vectors = normalized(documents) if documents else np.empty((0, dimension))
            self._vector_index = (index_key, entity_vectors, [row['provenance'] for row in documents])
        _, entity_vectors, provenance = self._vector_index
        similarities = dict(zip(entities, entity_vectors@query_vector))
        scores = [max(similarities[normalize(row['head'])], similarities[normalize(row['tail'])]) for row in self.triples]
        order = np.argsort(-np.asarray(scores), kind='stable')[:limit]
        return [self.triples[i] for i in order], [response['provenance']]+provenance


def extract(text, source, backend):
    response = backend.call('knowledge/extract', {'text':text,'source':source})
    graph = KnowledgeGraph(response['result']['triples'], source)
    return graph, response['provenance']
