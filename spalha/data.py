"""Well-disjoint preprocessing and lazy center-depth windows (Eqs. 1–2)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGS = ['GR', 'SP', 'RT', 'AC', 'DEN', 'CNL', 'CAL', 'PE']
CLASSES = ['Pure oil', 'Oil–water', 'Oil-bearing water', 'Pure water', 'Poor oil',
           'Weakly waterflooded', 'Moderately waterflooded', 'Strongly waterflooded']
CODES = {0: -1, **{2**i: i for i in range(8)}}


def read_frame(path):
    df = pd.read_csv(path, dtype={'well_id': str})
    required = ['well_id', 'depth', 'split', 'class_code', *LOGS]
    if any(k not in df for k in required):
        raise ValueError(f'Required CSV columns: {required}')
    if df.well_id.isna().any() or not np.isfinite(df.depth).all():
        raise ValueError('Well IDs and depths must be valid.')
    if df.groupby('well_id').split.nunique().max() != 1:
        raise ValueError('Well leakage: a well appears in multiple splits.')
    if not df.split.isin(['train', 'val', 'test', 'external']).all():
        raise ValueError('Invalid split.')
    if not df.class_code.isin(CODES).all():
        raise ValueError('Invalid class bit code.')
    for k in ('sw', 'phi'):
        if k not in df:
            df[k] = np.nan
        if not df[k].dropna().between(0, 1).all():
            raise ValueError(f'{k} must be a fraction in [0, 1].')
    # Only explicit, geologically verified comparison groups permit ranking.
    if 'rank_group' not in df:
        df['rank_group'] = ''
    return df.sort_values(['well_id', 'depth']).reset_index(drop=True)


def raw_values(df):
    raw = df[LOGS].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(raw)
    valid[:, 2] &= raw[:, 2] > 0
    valid[:, 4] &= raw[:, 4] > 0
    for j, name in enumerate(LOGS):
        if f'valid_{name}' in df:
            valid[:, j] &= df[f'valid_{name}'].fillna(0).to_numpy() == 1
    raw[~valid] = np.nan
    return raw, valid


def transform(raw):
    x = raw.copy()
    x[:, 2] = np.log1p(x[:, 2])
    return x


def fit_stats(df):
    # Table S1: training-only clipping and median/IQR normalization.
    x = transform(raw_values(df[df.split == 'train'])[0])
    if not len(x) or np.isnan(x).all(0).any():
        raise ValueError('Training logs must contain valid measurements in all channels.')
    lo, q1, med, q3, hi = np.nanpercentile(x, [0.5, 25, 50, 75, 99.5], axis=0)
    return {k: v.tolist() for k, v in dict(lo=lo, hi=hi, med=med, iqr=np.maximum(q3-q1, 1e-6)).items()}


class WellWindows(Dataset):
    def __init__(self, df, split, stats, window=33, stride=1, delta=0.125):
        if window < 3 or window % 2 != 1 or stride < 1:
            raise ValueError('Expected an odd window and a positive stride.')
        self.window = window
        self.half = window // 2
        self.wells, self.index = [], []
        for well, g in df[df.split == split].groupby('well_id', sort=True):
            depth = g.depth.to_numpy(dtype=np.float64)
            if np.any(np.diff(depth) <= 0):
                raise ValueError(f'Duplicate or unordered depths in {well}')
            # Split at gaps: no window crosses missing depth intervals or wells.
            cuts = np.r_[0, np.flatnonzero(~np.isclose(np.diff(depth), delta, atol=1e-5)) + 1, len(g)]
            for left, right in zip(cuts[:-1], cuts[1:]):
                seg = g.iloc[left:right]
                raw, valid = raw_values(seg)
                x = (np.clip(transform(raw), stats['lo'], stats['hi']) - stats['med']) / stats['iqr']
                x = np.where(valid, x, 0).astype(np.float32)
                record = dict(x=x, raw=np.nan_to_num(raw, nan=0.0), valid=valid.astype(np.float32),
                              depth=depth[left:right].astype(np.float32),
                              y=seg.class_code.map(CODES).to_numpy(dtype=np.int64),
                              sw=seg.sw.to_numpy(dtype=np.float32), phi=seg.phi.to_numpy(dtype=np.float32),
                              rank_group=seg.rank_group.fillna('').astype(str).to_numpy(), well=well)
                j = len(self.wells)
                self.wells.append(record)
                self.index.extend((j, i) for i in range(0, len(seg)-window+1, stride))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        j, start = self.index[idx]
        w = self.wells[j]
        sl = slice(start, start+self.window)
        c = start+self.half
        item = {k: torch.from_numpy(w[k][sl]) for k in ('x', 'raw', 'valid', 'depth')}
        item.update({k: torch.tensor(w[k][c]) for k in ('y', 'sw', 'phi')})
        item.update(well=w['well'], rank_group=w['rank_group'][c])
        return item
