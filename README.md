# SPaLHA-KSNOT

Semi-Supervised Physics-Anchored Local Hierarchical Attention KAN Sequential Neural Operator Transformer for complex oil–water reservoir identification.

## Architecture

1. Eight conventional logs, validity masks, and relative depth positions (Eqs. 1–2).
2. Cubic B-spline KAN embedding and bidirectional GRU encoding (Eq. 3).
3. Effective porosity, water saturation and shale volume estimation; density mixing and shale-weighted Archie residuals (Eqs. 4–5).
4. Residual-conditioned SE channel excitation (Eq. 6).
5. Stratigraphic connectivity, KAN query/key/value projections, multiscale local attention and iterative physics feedback (Eqs. 7–9).
6. Center-depth eight-class posterior decoding (Eq. 10).
7. EMA teacher–student consistency with confidence and physical-reliability filtering, robust physical residuals, boundary-aware continuity, and conditional waterflood ranking (Eqs. 11–12).

## Repository layout

```text
SPaLHA-KSNOT/
├── README.md
├── CHANGELOG.md
├── HYPERPARAMETERS.md
├── checkpoints/
│   ├── software_test_only.pt
│   └── software_test.log
├── tools/
│   ├── compare_weights.py
│   ├── software_test.py
│   └── tune.py
├── requirements.txt
├── configs/
│   ├── default.json
│   ├── candidate.json
│   ├── software_test.json
│   └── smoke.json
├── data/schema.csv
├── spalha/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── engine.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── tests/smoke_test.py
└── .github/workflows/ci.yml
```

## Installation

Python 3.10+ is recommended. Install a PyTorch build appropriate for your CUDA environment if GPU training is required.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/smoke_test.py -q
```

## Data preparation

Use a CSV with the header in `data/schema.csv`. Each row is a measured depth point. `well_id` must uniquely identify a physical well across all blocks. Assign complete wells to `train`, `val`, `test` or `external` before running the code. Do not use validation/test data to calibrate preprocessing or physical constants. The loader sorts depths, checks well leakage, and refuses to cross wells or sampling gaps. Measurements failing borehole/log quality checks should be marked missing (blank or optional `valid_GR`, ..., `valid_PE` flags, 1 = valid). The code does not invent quality-control thresholds.

| Column | Definition |
|---|---|
| depth | Depth in metres, nominal spacing 0.125 m |
| GR | API |
| SP | mV |
| RT | Raw resistivity in ohm m (not log-transformed) |
| AC | microseconds/m |
| DEN | g/cm³ |
| CNL | % |
| CAL | mm |
| PE | barn/electron |
| sw, phi | Optional independent observations in [0, 1]; blank if unavailable |
| class_code | 0 for unlabeled; one of the eight codes below otherwise |
| rank_group | Optional validated comparable lithologic/petrophysical group ID; blank disables ranking for that sample |

| Code | Reservoir class |
|---:|---|
| 1 | Pure oil layer |
| 2 | Oil–water layer |
| 4 | Oil-bearing water layer |
| 8 | Pure water layer |
| 16 | Poor oil layer |
| 32 | Weakly waterflooded layer |
| 64 | Moderately waterflooded layer |
| 128 | Strongly waterflooded layer |

The supplied manuscript describes 458 wells (320/69/69), 33-point windows, 0.125 m sampling, and a 25% labeled training subset. Use the actual experimental label mask, not a randomly regenerated mask. This pipeline does not infer the exact original window stride or force the manuscript's reported sample counts. It uses the supplied `class_code=0` mask. Preprocessing follows Supplementary Table S1: ln(1+RT), 0.5–99.5 percentile clipping and median/IQR scaling using training wells. Missing normalized values are set to zero with separate validity masks.

## Training

`configs/default.json` now contains exactly the previously reviewed candidate settings, with all numerical fields populated. It is identical to `configs/candidate.json`. The manuscript does not independently establish all settings or physical calibration constants; the author's confirmation is not a substitute for the original training records. Use the actual training-well calibration and label masks when validating the scientific results. The smoke configuration is for software tests only.

```bash
python train.py --config configs/default.json --data data/wells.csv --output runs/seed47
```

Validation Macro-F1 determines the best checkpoint. `history.jsonl`, `config.json` and `best.pt` are written to the output directory. To conduct five-seed runs, use separate configs with seeds 11, 29, 47, 71, 103 and separate output directories.
