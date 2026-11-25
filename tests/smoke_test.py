"""Synthetic tensor checks only; these values are not experimental data."""
import json
from pathlib import Path
import torch
from spalha import SPaLHAKSNOT
from spalha.losses import objective, ConfidenceThresholds, augment


def test_forward_backward():
    torch.set_num_threads(1)
    cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/smoke.json').read_text())
    model = SPaLHAKSNOT(cfg['model'])
    x = torch.randn(4,33,8)
    raw = x.abs()+1
    raw[...,4] = 2.4
    b = dict(x=x,raw=raw,valid=torch.ones_like(x),depth=torch.arange(33)[None].repeat(4,1)*0.125,
             y=torch.tensor([5,6,7,-1]),sw=torch.tensor([0.4,0.5,0.8,float('nan')]),
             phi=torch.tensor([0.2,0.2,0.1,float('nan')]),rank_group=['g','g','g',''])
    with torch.no_grad():
        teacher = model(b)
    out = model(augment(b,**cfg['strong_augmentation']))
    assert out['logits'].shape == (4,33,8)
    assert torch.allclose(out['attention'].sum(-1),torch.ones_like(out['attention'].sum(-1)),atol=1e-5)
    assert ((out['sw']>=0)&(out['sw']<=1)).all()
    loss,_ = objective(out,teacher,b,cfg['loss'],ConfidenceThresholds(0.9,'cpu'))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


if __name__ == '__main__':
    test_forward_backward()
    print('Forward/backward smoke test passed.')
