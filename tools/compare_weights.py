"""Compare tensor names, shapes and numerical differences without changing weights."""
import argparse
import json
from pathlib import Path
import torch


def load(path):
    # Only use trusted tensor-only checkpoint files; no arbitrary object loading.
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError('Expected a dictionary checkpoint.')
    for key in ('model','state_dict','model_state_dict'):
        if key in obj and isinstance(obj[key], dict):
            return obj[key], obj.get('provenance', {})
    if all(torch.is_tensor(v) for v in obj.values()):
        return obj, {}
    raise ValueError('No supported model state dictionary found.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate'); p.add_argument('original')
    p.add_argument('--output',default='weight_comparison.json')
    a=p.parse_args()
    left, lp = load(a.candidate); right, rp = load(a.original)
    matched=[]; incompatible=[]
    for name in sorted(left.keys() & right.keys()):
        x,y=left[name],right[name]
        if x.shape!=y.shape:
            incompatible.append(dict(name=name,candidate_shape=list(x.shape),original_shape=list(y.shape)))
        else:
            diff=(x.double()-y.double()).abs()
            matched.append(dict(name=name,shape=list(x.shape),identical=bool(torch.equal(x,y)),max_abs_difference=float(diff.max()) if diff.numel() else 0.0))
    report=dict(candidate_provenance=lp,original_provenance=rp,matched=matched,shape_mismatches=incompatible,
                only_candidate=sorted(left.keys()-right.keys()),only_original=sorted(right.keys()-left.keys()))
    report['strict_shape_compatible']=not any(report[k] for k in ('shape_mismatches','only_candidate','only_original'))
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(f"Matching shapes: {len(matched)}; mismatches: {len(incompatible)}; strict compatibility: {report['strict_shape_compatible']}")


if __name__=='__main__':
    main()
