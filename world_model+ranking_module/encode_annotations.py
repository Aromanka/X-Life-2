"""Extract structured annotations through the external role service, supporting API and local models behind the role service."""
import argparse
import json
from pathlib import Path
from backends.http import HTTPBackend
from configuration import parse_configured


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--backend-url')
    p.add_argument('--role',choices=['teacher','action/explicit'])
    p.add_argument('--dimension',type=int)
    args=parse_configured(p, argv, 'annotation', ('role', 'dimension'))
    records=json.loads(Path(args.input).read_text(encoding='utf-8'))
    if not isinstance(records,list): raise ValueError('Input must be a list of explicitly selected observations')
    backend=HTTPBackend(args.backend_url);results=[]
    for record in records:
        response=backend.call(args.role,dict(observation=record,action=record,text=json.dumps(record,ensure_ascii=False),dimension=args.dimension))
        results.append(response)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'records':len(results),'test_only':any(r['provenance']['test_backend'] for r in results)}))


if __name__=='__main__': main()
