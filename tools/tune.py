"""Run a small candidate search using real training and validation wells only."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='Configuration with independently calibrated physical constants')
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args(); base=json.loads(Path(a.config).read_text()); out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if any(v is None for v in base['model']['physics'].values()):
        raise ValueError('Provide training-only calibrated physical constants first.')
    reports=[]
    # Candidate values are a search space, not reported experimental settings.
    for idx,(hidden,lr,weight) in enumerate(itertools.product([96,128],[0.0003,0.001],[0.1,0.3])):
        c=copy.deepcopy(base); c['model']['hidden']=hidden; c['learning_rate']=lr; c['loss']['weights']['physics']=weight; c['epochs']=a.epochs
        d=out/f'trial_{idx:02d}'; d.mkdir(exist_ok=False); path=d/'trial_config.json'; path.write_text(json.dumps(c,indent=2))
        subprocess.run([sys.executable,'train.py','--config',str(path),'--data',a.data,'--output',str(d),'--device',a.device],check=True)
        history=[json.loads(line) for line in (d/'history.jsonl').read_text().splitlines()]
        best=max(history,key=lambda r:r['macro_f1']); reports.append(dict(trial=idx,hidden=hidden,learning_rate=lr,physics_weight=weight,validation_macro_f1=best['macro_f1'],checkpoint=str(d/'best.pt')))
    reports.sort(key=lambda r:r['validation_macro_f1'],reverse=True)
    (out/'validation_ranking.json').write_text(json.dumps(reports,indent=2)); print(json.dumps(reports[0],indent=2))


if __name__=='__main__':
    main()
