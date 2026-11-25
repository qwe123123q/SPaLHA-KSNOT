"""Semi-supervised, physics, structural and conditional ranking objectives."""
import torch
from torch.nn import functional as F


def center(t):
    return t[:, t.size(1)//2]


def augment(batch, noise, amplitude, drop):
    """Preserve depth and labels; perturb valid measurements only.

    Noise is applied to the normalized feature branch; physical residuals
    retain the original measurements except for randomly masked channels.
    """
    result = dict(batch)
    x, valid = batch['x'], batch['valid']
    mask = valid * (torch.rand_like(valid) >= drop)
    scale = 1 + amplitude * torch.randn_like(x[:, :1])
    result['x'] = (x * scale + noise * torch.randn_like(x)) * mask
    result['valid'] = mask
    return result


class ConfidenceThresholds:
    """Training-only EMA of class-specific labeled teacher confidence."""
    def __init__(self, initial, device, momentum=0.95):
        self.values = torch.full((8,), initial, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, p, y):
        conf, pred = p.max(-1)
        for k in range(8):
            good = (y == k) & (pred == k)
            if good.any():
                target = conf[good].mean().clamp(0.70, 0.99)
                self.values[k].lerp_(target, 1-self.momentum)


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(decay).add_(s, alpha=1-decay)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)


def objective(out, teacher, batch, cfg, thresholds):
    """Eqs. 11–12 plus explicit masked auxiliary regression supervision."""
    logits = center(out['logits'])
    y = batch['y']
    zero = logits.sum() * 0
    labeled = y >= 0
    supervised = F.cross_entropy(logits[labeled], y[labeled]) if labeled.any() else zero
    with torch.no_grad():
        p = center(teacher['logits']).softmax(-1)
        thresholds.update(p, y)
        conf, pred = p.max(-1)
        reliability = center(teacher['feedback']).norm(dim=-1)
        available = center(teacher['weights']).sum(-1) > 0
        selected = (~labeled) & available & (conf >= thresholds.values[pred]) & (reliability <= cfg['residual_threshold'])
    consistency = F.kl_div(logits[selected].log_softmax(-1), p[selected], reduction='batchmean') if selected.any() else zero
    w = out['weights']
    physics = (F.huber_loss(out['residual'], torch.zeros_like(w), reduction='none') * w).sum() / w.sum().clamp_min(1)
    probs = out['logits'].softmax(-1)
    # Stop gradients through boundary weights in this penalty to avoid a trivial all-boundary solution.
    structural = ((probs[:, 1:] - probs[:, :-1]).square().sum(-1) * (1-out['boundary'].detach())).mean()
    # Compare only verified groups and different waterflood stages, never all eight classes.
    pp = logits.softmax(-1)[:, 5:8]
    score = (pp * pp.new_tensor([0., 1., 2.])).sum(-1)
    groups = batch['rank_group']
    pair = torch.tensor([[bool(a) and a == b for b in groups] for a in groups], device=y.device)
    pair &= (y[:, None] >= 5) & (y[None, :] >= 5) & (y[:, None] > y[None, :])
    ranking = F.relu(cfg['rank_margin'] - (score[:, None] - score[None, :]))[pair].mean() if pair.any() else zero
    auxiliary = zero
    for k in ('phi', 'sw'):
        target = batch[k]
        observed = torch.isfinite(target)
        if observed.any():
            auxiliary = auxiliary + F.smooth_l1_loss(center(out[k])[observed], target[observed])
    terms = dict(supervised=supervised, consistency=consistency, physics=physics,
                 structural=structural, ranking=ranking, auxiliary=auxiliary)
    total = supervised + sum(cfg['weights'][k]*terms[k] for k in terms if k != 'supervised')
    return total, {**{k: float(v.detach()) for k, v in terms.items()}, 'selected': int(selected.sum())}
