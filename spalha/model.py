"""Physics-anchored sequential operator reconstructed from manuscript Eqs. (3–10)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    """SiLU base mapping plus learnable cubic B-spline expansion."""
    def __init__(self, inputs, outputs, intervals=5, degree=3):
        super().__init__()
        self.degree = degree
        self.register_buffer('knots', torch.arange(-degree, intervals + degree + 1).float() * 6 / intervals - 3)
        self.base = nn.Linear(inputs, outputs)
        self.coeff = nn.Parameter(torch.empty(outputs, inputs, intervals + degree))
        nn.init.normal_(self.coeff, std=0.02 / math.sqrt(inputs))

    def forward(self, x):
        # Bound the spline argument; retain the unbounded SiLU base branch.
        u = x.clamp(-3, 3 - 1e-5).unsqueeze(-1)
        t = self.knots
        b = ((u >= t[:-1]) & (u < t[1:])).to(x.dtype)
        for k in range(1, self.degree + 1):
            b = ((u - t[:-(k+1)]) / (t[k:-1] - t[:-(k+1)])) * b[..., :-1] + ((t[k+1:] - u) / (t[k+1:] - t[1:-k])) * b[..., 1:]
        return self.base(F.silu(x)) + F.linear(b.flatten(-2), self.coeff.flatten(1))


class Petrophysics(nn.Module):
    """Density mixing and shale-weighted Archie residuals (Eqs. 4–5).

    All constants must be calibrated using training wells only.
    Raw RT (ohm m) and DEN (g/cm^3) are never standardized here.
    """
    def __init__(self, hidden, constants):
        super().__init__()
        required = ('phi_max', 'rho_ma', 'rho_w', 'rho_o', 'rw', 'a', 'm', 'n', 's_rho', 's_rt', 'gamma')
        if any(constants.get(k) is None for k in required):
            raise ValueError('Supply training-well calibrated petrophysical constants.')
        for k in required:
            v = float(constants[k])
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f'Invalid physical constant: {k}')
            self.register_buffer(k, torch.tensor(v))
        if self.phi_max > 1:
            raise ValueError('Porosity must be expressed as a fraction.')
        self.head = nn.Linear(hidden, 3)

    def forward(self, h, raw, valid):
        p = torch.sigmoid(self.head(h))
        phi, sw, shale = p.unbind(-1)
        phi = phi * self.phi_max
        density = (1 - phi) * self.rho_ma + phi * (sw * self.rho_w + (1 - sw) * self.rho_o)
        log_rt = torch.log(self.a * self.rw) - self.m * torch.log(phi + 1e-6) - self.n * torch.log(sw + 1e-6)
        r = torch.stack(((raw[..., 4] - density) / self.s_rho,
                         (raw[..., 2].clamp_min(1e-6).log() - log_rt) / self.s_rt), -1)
        weights = torch.stack((valid[..., 4], valid[..., 2] * (1 - shale).pow(self.gamma)), -1)
        return {'phi': phi, 'sw': sw, 'shale': shale, 'residual': r,
                'weights': weights, 'feedback': r * weights}


class CoupledOperator(nn.Module):
    """Physics-conditioned SE, boundary connectivity and multiscale KAN attention."""
    def __init__(self, hidden, radii, intervals, residual_penalty):
        super().__init__()
        self.radii = radii
        self.penalty = residual_penalty
        self.se = nn.Sequential(nn.Linear(hidden, max(1, hidden // 4)), nn.SiLU(), nn.Linear(max(1, hidden // 4), hidden), nn.Sigmoid())
        self.gate = nn.Linear(hidden + 2, hidden)
        self.boundary = nn.Linear(hidden + 2, 1)
        self.qkv = nn.ModuleList([nn.ModuleList([KANLinear(hidden, hidden, intervals) for _ in range(3)]) for _ in radii])
        self.fusion = nn.Linear(hidden + 3, len(radii))
        self.update = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, x, valid, r, depth):
        # Eq. 6: recompute the gate after every petrophysical update.
        g = self.se(h.mean(1))[:, None] * torch.sigmoid(self.gate(torch.cat((h, r), -1)))
        # Eq. 7: latent, GR and logarithmic RT jumps; missing logs do not create jumps.
        jumps = (x[:, 1:, [0, 2]] - x[:, :-1, [0, 2]]).abs()
        jumps = jumps * valid[:, 1:, [0, 2]] * valid[:, :-1, [0, 2]]
        beta = torch.sigmoid(self.boundary(torch.cat(((h[:, 1:] - h[:, :-1]).abs(), jumps), -1))).squeeze(-1)
        # Log-space product of (1 - beta) over every intervening edge.
        prefix = F.pad(torch.log1p(-beta.clamp(max=1-1e-6)).cumsum(-1), (1, 0))
        log_conn = -(prefix[:, :, None] - prefix[:, None, :]).abs()
        distance = torch.arange(h.size(1), device=h.device)
        distance = (distance[:, None] - distance[None, :]).abs()
        residual_dist = (r[:, :, None] - r[:, None, :]).square().sum(-1)
        dz = depth[:, 1:] - depth[:, :-1]
        quadrature = torch.cat((dz[:, :1]/2, (dz[:, :-1]+dz[:, 1:])/2, dz[:, -1:]/2), -1)
        outputs, kernels = [], []
        for radius, (q, k, v) in zip(self.radii, self.qkv):
            # Eq. 8: physically penalized local integration kernel.
            score = q(h) @ k(h).transpose(-1, -2) / math.sqrt(h.size(-1))
            score = score - self.penalty * residual_dist + log_conn + quadrature.clamp_min(1e-6).log()[:, None, :]
            attn = score.masked_fill(distance > radius, -torch.inf).softmax(-1)
            outputs.append(attn @ (g * v(h)))
            kernels.append(attn)
        # Eq. 9: dynamic scale fusion and recurrent state update.
        local_boundary = torch.maximum(F.pad(beta, (1, 0)), F.pad(beta, (0, 1)))
        alpha = self.fusion(torch.cat((h, r, local_boundary[..., None]), -1)).softmax(-1)
        mixed = (torch.stack(outputs, -2) * alpha[..., None]).sum(-2)
        return self.norm(h + self.update(mixed)), beta, alpha, torch.stack(kernels, 1)


class SPaLHAKSNOT(nn.Module):
    """Eight-class classification with Sw, effective porosity and shale outputs."""
    def __init__(self, cfg):
        super().__init__()
        h = cfg['hidden']
        if h % 2 or cfg['iterations'] < 1:
            raise ValueError('hidden must be even and iterations positive')
        self.embedding = KANLinear(17, h, cfg['spline_intervals'])
        self.encoder = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        self.physics = Petrophysics(h, cfg['physics'])
        self.operator = CoupledOperator(h, cfg['radii'], cfg['spline_intervals'], cfg['residual_penalty'])
        self.iterations = cfg['iterations']
        self.decoder = KANLinear(h, 8, cfg['spline_intervals'])

    def forward(self, batch):
        x, raw, mask, depth = (batch[k] for k in ('x', 'raw', 'valid', 'depth'))
        relative = (depth - depth[:, depth.size(1)//2:depth.size(1)//2+1]) / 2.0
        h, _ = self.encoder(self.embedding(torch.cat((x, mask, relative[..., None]), -1)))
        for _ in range(self.iterations):
            physics = self.physics(h, raw, mask)
            h, beta, alpha, attention = self.operator(h, x, mask, physics['feedback'], depth)
        return dict(self.physics(h, raw, mask), logits=self.decoder(h), boundary=beta,
                    hierarchy=alpha, attention=attention)
