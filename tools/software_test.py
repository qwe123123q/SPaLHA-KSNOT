"""Generate a tiny deterministic software-only fixture and exercise the full-width model.

The synthetic fixture is not a well dataset and must never be used as paper evidence.
The saved checkpoint is solely for serialization and compatibility tests.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fixture(path):
    """Create finite tensors in the input schema, not simulated field observations."""
    rng = np.random.default_rng(47)
    rows = []
    for split, n_wells in [('train', 4), ('val', 2), ('test', 2)]:
        for well in range(n_wells):
            for i in range(40):
                k = (i + 2 * well) % 8
                logs = [80, -12, 20, 320, 2.35, 20, 220, 3.0]
                noise = rng.normal(size=8) * [8, 3, 2, 8, .03, 2, 3, .2]
                record = dict(zip(['GR', 'SP', 'RT', 'AC', 'DEN', 'CNL', 'CAL', 'PE'], np.array(logs)+noise))
                record.update(well_id=f'SOFTWARE_TEST_{split}_{well}', split=split, depth=1000+i*.125,
                              class_code=0 if split=='train' and (i+well)%4 else 2**k,
                              sw=.2+.07*k, phi=.15+.01*k, rank_group='')
                rows.append(record)
    pd.DataFrame(rows).to_csv(path,index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/software_test')
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = out / 'software_fixture.csv'
    fixture(data)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, 'train.py', '--config', 'configs/software_test.json', '--data', str(data), '--output', str(out), '--device','cpu'],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(out/'best.pt'), '--data',str(data),'--split','test','--output',str(out/'evaluation'),'--device','cpu'],cwd=ROOT,env=env,check=True)
    # Save explicit provenance in the checkpoint itself, not just its filename.
    import torch
    ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    ckpt['provenance']={'kind':'synthetic_software_test','original_experiment':False,'seed':47,'purpose':'serialization and numerical smoke testing only'}
    target=ROOT/'checkpoints/software_test_only.pt'
    torch.save(ckpt,target)
    print(f'Wrote {target}; this is NOT an original experimental checkpoint.')


if __name__=='__main__':
    main()
