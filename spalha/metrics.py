"""Metrics calculated from predictions, never from manuscript target scores."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, r2_score, confusion_matrix


def evaluate_frame(df, delta=0.125, stride=1, bins=15):
    """Compute exact adjacent-center boundary F1; no cross-well edges."""
    d = df[df.true_class >= 0].copy()
    if not len(d):
        return {'labeled_count': 0}
    y, p = d.true_class.to_numpy(), d.pred_class.to_numpy()
    conf = d.confidence.to_numpy()
    ece = 0.
    for i in range(bins):
        mask = (conf >= i/bins) & ((conf < (i+1)/bins) if i < bins-1 else (conf <= 1))
        if mask.any():
            ece += mask.mean()*abs((y[mask] == p[mask]).mean()-conf[mask].mean())
    actual, predicted = [], []
    for _, g in d.groupby('well_id'):
        g = g.sort_values('depth')
        ok = np.isclose(np.diff(g.depth), delta*stride, atol=1e-4)
        actual.extend((np.diff(g.true_class) != 0)[ok])
        predicted.extend((np.diff(g.pred_class) != 0)[ok])
    result = {'labeled_count': len(d), 'accuracy': float(accuracy_score(y,p)),
              'macro_f1': float(f1_score(y,p,labels=list(range(8)),average='macro',zero_division=0)),
              'kappa': float(cohen_kappa_score(y,p)), 'ece': float(ece),
              'boundary_f1': float(f1_score(actual,predicted,zero_division=0)) if actual else None,
              'confusion_matrix': confusion_matrix(y,p,labels=list(range(8))).tolist()}
    for k in ('sw','phi'):
        valid = df[f'true_{k}'].notna()
        result[f'{k}_r2'] = float(r2_score(df.loc[valid,f'true_{k}'],df.loc[valid,f'pred_{k}'])) if valid.sum() > 1 else None
    return result
