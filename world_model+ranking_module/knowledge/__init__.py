"""Local graph snapshots with separate generation and safety retrieval strategies."""
import json
import re
import unicodedata
from pathlib import Path


def normalize(value):
    return unicodedata.normalize('NFKC', str(value)).strip().casefold()


class KnowledgeGraph:
    def __init__(self, triples=(), source='explicit-empty', provenance=()):
        self.source = source
        self.provenance=list(provenance)
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

    def retrieve(self, terms, strategy='generation', limit=8):
        terms = [normalize(t) for t in terms if t]
        if strategy == 'safety':
            indices = set().union(*(self.index.get(t,set()) for t in terms)) if terms else set()
        elif strategy == 'generation':
            indices = {i for i,row in enumerate(self.triples) if any(t in normalize(row['head']+' '+row['tail']) for t in terms)}
        else: raise ValueError('Unknown retrieval strategy')
        return [self.triples[i] for i in sorted(indices)[:limit]]

    def retrieve_vector(self, query, backend, dimension=128, limit=8):
        import numpy as np
        responses = [backend.call('embed', {'text':text, 'dimension':dimension}) for text in
                     [query]+[r['head']+' '+r['relation']+' '+r['tail'] for r in self.triples]]
        vectors = np.asarray([r['result']['vector'] for r in responses], dtype=float)
        if vectors.shape != (len(responses),dimension) or not np.isfinite(vectors).all(): raise ValueError('Invalid embeddings')
        vectors /= np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
        order = np.argsort(-(vectors[1:]@vectors[0]))[:limit]
        return [self.triples[i] for i in order], [r['provenance'] for r in responses]


def extract(text, source, backend):
    response = backend.call('knowledge/extract', {'text':text,'source':source})
    graph = KnowledgeGraph(response['result']['triples'], source)
    return graph, response['provenance']
