"""Train SPaLHA-KSNOT with well-level validation and EMA pseudo-labels."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from spalha import SPaLHAKSNOT
from spalha.data import read_frame, fit_stats, WellWindows
from spalha.losses import ConfidenceThresholds, augment, objective, update_teacher
from spalha.engine import seed_all, to_device, predict
from spalha.metrics import evaluate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    seed_all(cfg['seed'])
    device = torch.device(args.device)
    df = read_frame(args.data)
    stats = fit_stats(df)
    train, val = [WellWindows(df,s,stats,**cfg['window']) for s in ('train','val')]
    if not len(train) or not len(val):
        raise ValueError('Train and validation splits must contain complete windows.')
    train_loader = DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=0)
    val_loader = DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    model = SPaLHAKSNOT(cfg['model']).to(device)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    thresholds = ConfidenceThresholds(cfg['confidence_threshold'],device)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; windows: train={len(train)}, val={len(val)}')
    best, stale = -1., 0
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        running, selected = 0., 0
        for batch in train_loader:
            batch = to_device(batch,device)
            with torch.no_grad():
                target = teacher(augment(batch,**cfg['weak_augmentation']))
            out = model(augment(batch,**cfg['strong_augmentation']))
            loss, terms = objective(out,target,batch,cfg['loss'],thresholds)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            update_teacher(teacher,model,cfg['ema_decay'])
            running += loss.item()
            selected += terms['selected']
        # Model selection is restricted to validation wells, never test wells.
        metrics = evaluate_frame(predict(model,val_loader,device),cfg['window']['delta'],cfg['window']['stride'])
        if 'macro_f1' not in metrics:
            raise ValueError('Validation requires labeled center points.')
        record = dict(epoch=epoch,loss=running/len(train_loader),pseudo_labels=selected,**metrics)
        print(json.dumps(record))
        with (output/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            torch.save(dict(model=model.state_dict(),teacher=teacher.state_dict(),config=cfg,stats=stats,epoch=epoch),output/'best.pt')
        else:
            stale += 1
            if stale >= cfg['patience']:
                break


if __name__ == '__main__':
    main()
