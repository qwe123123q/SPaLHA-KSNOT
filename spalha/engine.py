"""Reproducible training, checkpointing, and per-depth inference utilities."""
import random
import numpy as np
import pandas as pd
import torch
from .losses import center


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    frames = []
    for b in loader:
        b = to_device(b, device)
        out = model(b)
        p = center(out['logits']).softmax(-1)
        confidence, pred = p.max(-1)
        arr = lambda t: t.detach().cpu().numpy()
        data = dict(well_id=b['well'], depth=arr(center(b['depth'])), true_class=arr(b['y']),
                    pred_class=arr(pred), pred_code=arr(2**pred), confidence=arr(confidence),
                    true_sw=arr(b['sw']), pred_sw=arr(center(out['sw'])), true_phi=arr(b['phi']),
                    pred_phi=arr(center(out['phi'])), pred_shale=arr(center(out['shale'])),
                    density_residual=arr(center(out['residual'])[:,0]), rt_residual=arr(center(out['residual'])[:,1]))
        data.update({f'p_{i}': arr(p[:,i]) for i in range(8)})
        frames.append(pd.DataFrame(data))
    if not frames:
        raise ValueError('No complete windows in the requested split.')
    return pd.concat(frames, ignore_index=True).sort_values(['well_id','depth'])
