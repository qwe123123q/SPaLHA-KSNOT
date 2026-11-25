# Hyperparameter verification sheet — v1.0.0

Source: supplied manuscript and Supplementary Tables S1–S3. The preceding candidate values are frozen in default.json following the user’s confirmation. The status column preserves their original evidentiary provenance: values not documented in the manuscript have not been independently verified against original training logs. Original experimental weights were not provided.

| Parameter | Candidate | Status |
|---|---:|---|
| Log channels | 8 | Documented |
| Reservoir classes | 8 | Documented |
| Depth window | 33 | Documented |
| Sampling interval (m) | 0.125 | Documented |
| Train/validation/test wells | 320/69/69 | Documented |
| Labeled training fraction | 0.25 | Documented; actual label mask required |
| Seeds | 11, 29, 47, 71, 103 | Documented |
| Clipping percentiles | 0.5, 99.5 | Documented |
| Resistivity transform | ln(1+RT) | Documented |
| Normalization | Median–IQR | Documented in Table S1; equation description should be checked |
| Window stride | 1 | Provisional |
| Latent width | 128 | Provisional |
| Feedback iterations | 3 | Provisional |
| Attention radii (samples) | 2, 4, 8 | Provisional |
| Spline intervals / degree | 5 / 3 | Provisional |
| Residual penalty | 0.5 | Provisional |
| Batch size | 32 | Provisional |
| Optimizer | AdamW | Provisional |
| Learning rate | 0.001 | Provisional |
| Weight decay | 0.0001 | Provisional |
| Epochs / patience | 100 / 15 | Provisional |
| Gradient clipping | 1.0 | Provisional |
| Teacher EMA | 0.99 | Provisional |
| Initial confidence | 0.90 | Provisional |
| Confidence EMA / bounds | 0.95 / 0.70–0.99 | Provisional |
| Residual threshold | 2.0 | Provisional |
| Rank margin | 0.1 | Provisional |
| Weak noise / amplitude / drop | 0.01 / 0.01 / 0.02 | Provisional |
| Strong noise / amplitude / drop | 0.05 / 0.05 / 0.10 | Provisional |
| Consistency loss weight | 1.0 | Provisional |
| Physics loss weight | 0.1 | Provisional |
| Structural loss weight | 0.1 | Provisional |
| Ranking loss weight | 0.1 | Provisional |
| Auxiliary regression weight | 1.0 | Provisional |
| Maximum porosity | 0.40 | Uncalibrated |
| Matrix density (g/cm³) | 2.65 | Uncalibrated |
| Water density (g/cm³) | 1.00 | Uncalibrated |
| Oil density (g/cm³) | 0.80 | Uncalibrated |
| Formation-water resistivity (Ω m) | 0.10 | Uncalibrated |
| Archie a, m, n | 1, 2, 2 | Uncalibrated |
| Density residual scale | 0.10 | Uncalibrated |
| Log-resistivity residual scale | 1.00 | Uncalibrated |
| Shale attenuation exponent | 2.00 | Uncalibrated |
| Trainable parameters | 1,474,490 | Measured reconstruction; manuscript reports 7.86 M |

## Software validation

CPU, PyTorch 2.10.0: forward/backward test passed, three-epoch synthetic fixture training and evaluation completed, checkpoint self-comparison matched 79 state-dictionary tensors/buffers. These checks are not scientific validation or a reconstruction of original weights. No actual geological data, published predictions or original checkpoint are included.
