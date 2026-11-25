"""Evaluate a saved model on held-out wells or an external block."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from spalha import SPaLHAKSNOT
from spalha.data import read_frame, WellWindows
from spalha.engine import predict
from spalha.metrics import evaluate_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--split',choices=['val','test','external'],default='test')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)
    ckpt = torch.load(a.checkpoint,map_location=device,weights_only=True)
    cfg = ckpt['config']
    model = SPaLHAKSNOT(cfg['model']).to(device)
    model.load_state_dict(ckpt['model'])
    dataset = WellWindows(read_frame(a.data),a.split,ckpt['stats'],**cfg['window'])
    df = predict(model,DataLoader(dataset,batch_size=cfg['batch_size']),device)
    dest = Path(a.output)
    dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'predictions.csv',index=False)
    metrics = evaluate_frame(df,cfg['window']['delta'],cfg['window']['stride'])
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
