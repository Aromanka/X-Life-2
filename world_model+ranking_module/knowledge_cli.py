"""Build an attributed graph snapshot through HTTP extraction, then normalize/deduplicate."""
import argparse
import json
from pathlib import Path
from backends.http import HTTPBackend
from knowledge import KnowledgeGraph,extract,normalize
from configuration import parse_configured


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backend-url');p.add_argument('--chunk-chars',type=int)
    args=parse_configured(p, argv, 'knowledge', ('chunk_chars',))
    if args.chunk_chars<1: p.error('Positive chunk size required')
    backend=HTTPBackend(args.backend_url);triples={};provenance=[]
    for name in args.input:
        path=Path(name)
        if path.suffix.lower()=='.pdf':
            import pymupdf
            with pymupdf.open(path) as doc: text='\n'.join(page.get_text() for page in doc)
        else: text=path.read_text(encoding='utf-8')
        for offset in range(0,len(text),args.chunk_chars):
            graph,source=extract(text[offset:offset+args.chunk_chars],f'{path.name}:chars:{offset}',backend)
            provenance.append(source)
            for row in graph.triples:
                key=tuple(normalize(row[k]) for k in ('head','relation','tail','source'))
                triples[key]=row
    graph=KnowledgeGraph(triples.values(),'http-extraction')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(graph.triples,ensure_ascii=False,indent=2),encoding='utf-8')
    path.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'triples':len(graph.triples),'test_only':any(r['test_backend'] for r in provenance)}))


if __name__=='__main__': main()
