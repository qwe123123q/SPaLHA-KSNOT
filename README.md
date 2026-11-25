# SPaLHA-KSNOT
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

| 16 | Poor oil layer |
| 32 | Weakly waterflooded layer |
| 64 | Moderately waterflooded layer |
| 128 | Strongly waterflooded layer |
