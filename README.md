
# Parametric Curve Estimation Repository

All parameter estimation steps, derivations, plots, and code are written by me based on standard mathematical and signal-processing techniques. External concepts such as PCA, exponential envelopes, and L1 optimization are referenced appropriately in the References section in report. No code or text was copied from any student or online assignment source.

## How to Run:

```
py -m pip install matplotlib numpy pandas
py src/fit_params.py --data data/xy_data.csv

```

## Full Parameter Estimation (Automatic)

To re-estimate theta , M, and X from raw data:

py -m pip install matplotlib numpy pandas scipy
py src/full_estimation.py --data data/xy_data.csv --savefigs

This will:
- Estimate initial values using PCA + envelope analysis
- Refine parameters using L1 minimization
- Save plots into `figures/`
- Update `params.json`



