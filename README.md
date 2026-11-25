# SPaLHA-KSNOT

Semi-Supervised Physics-Anchored Local Hierarchical Attention KAN Sequential Neural Operator Transformer for complex oil–water reservoir identification.

**Release v1.0.0 — author-reviewed code package.** This release freezes the preceding implementation and configuration following the user's confirmation of consistency. It implements the core on-grid architecture and training pipeline described in the manuscript. Original well logs and original experimental checkpoints are not included. The included checkpoint is solely a software-test artifact. No published performance values are generated or hard-coded.

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

Validation Macro-F1 determines the best checkpoint. `history.jsonl`, `config.json` and `best.pt` are written to the output directory. To conduct five-seed runs, use separate configs with seeds 11, 29, 47, 71, 103 (reported in the manuscript) and separate output directories.

## Evaluation

```bash
python evaluate.py --checkpoint runs/seed47/best.pt --data data/wells.csv --split test --output runs/test
python evaluate.py --checkpoint runs/seed47/best.pt --data data/GTZKL.csv --split external --output runs/GTZKL
```

The checkpoint's training normalization is reused. `predictions.csv` contains center depths, class codes, eight posteriors, true and predicted properties where available, shale estimates, and physical residuals. `metrics.json` contains measured Accuracy, Macro-F1, Kappa, ECE, exact adjacent-center Boundary-F1 and regression R². Classification and calibration metrics are fractions. Boundary tolerance and physical violation-rate thresholds are not established in the supplied text; the implementation uses exact adjacent-center transitions and does not report PVR. External files must use the same schema and well IDs disjoint from training.

## Implementation boundaries

- The supplied text describes continuous query depths; this reconstruction implements the measured-grid center-depth task only. Arbitrary off-grid operator queries are not implemented.
- The architecture uses shared feedback weights, a residual update with LayerNorm, fixed spline knots, a class-confidence EMA and explicit masked auxiliary regression losses. These are documented implementation choices where original source code is absent. Structural consistency uses a detached boundary weighting to prevent a trivial all-boundary solution.
- Ranking requires explicit verified comparison groups. Random minibatches may contain few eligible pairs; use actual comparable-group metadata and consider group-aware sampling when reproducing experiments.
- The physical branch preserves raw measured values during feature augmentation; random masking removes their applicability. Actual measurement QC and calibrated physical constants remain the author's responsibility.
- The original comparison models, full ablation suite, performance timing and paper figure generation are not included. Exact parameter count and reported metrics are not guaranteed to match the manuscript.
- Tests use synthetic tensors solely to check executable behavior and numerical stability. They are not geological evidence.

## Availability and citation

Before depositing this repository as the code used to produce the paper, verify the remaining parameter-count and checkpoint discrepancies, provide the authorized data-access statement, and archive the verified release with a persistent DOI. Add the final article citation and choose a software license only after confirming author and institutional permissions. This repository has not been published to GitHub or Zenodo.

## Release configuration and checkpoint verification

`configs/default.json` and `configs/candidate.json` contain the same author-reviewed numerical settings. The latter is retained for provenance and backward compatibility. The constants were not calibrated against SGH data during development of this repository; original calibration records are required to substantiate their experimental provenance. `configs/software_test.json` runs the same width-128 architecture for three epochs on a tiny synthetic software fixture. The checkpoint `checkpoints/software_test_only.pt` is trained on that fixture and has explicit provenance metadata; it is NOT the original trained weights. The reconstructed architecture has 1,474,490 trainable parameters, whereas the manuscript reports approximately 7.86 million; exact architectural recovery requires the original model definition.

```bash
# Re-run the software test on CPU
python tools/software_test.py
# Compare actual tensor names, shapes, and numerical values with your own checkpoint
python tools/compare_weights.py checkpoints/software_test_only.pt /path/to/your_original.pt --output comparison.json
# After independently calibrating the physical constants, search on real training/validation wells only
python tools/tune.py --config configs/default.json --data data/wells.csv --output runs/tuning --device cuda
```

The search considers width, learning rate and physical loss weight; it selects configurations by validation Macro-F1, never test accuracy. It has not been run on the 458-well dataset. Other provisional hyperparameters are listed in `HYPERPARAMETERS.md`. The software test selected zero pseudo-labels in its short full-width run; this is not evidence of successful semi-supervised learning. `tools/compare_weights.py` checks tensor compatibility rather than claiming identity of different checkpoints.
