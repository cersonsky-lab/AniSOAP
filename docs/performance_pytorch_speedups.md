# AniSOAP PyTorch performance summary

This document summarizes timing results for the PyTorch backend compared to the original NumPy implementation. The full raw metrics and figures are stored in `benchmarks/ospo_pytorch_speedups/`.

Benchmarks cover:
- Benzene systems with varying numbers of molecules.
- Ellipsoid systems.
- Multi-species systems (one, three, and four species).

For each system, we compare:
- NumPy (CPU)
- PyTorch (CPU)
- PyTorch (Apple MPS, where available)

The associated CSV files are:

- `timings.csv`: wall-clock timings for each configuration.
- `combined_from_metrics.csv`: combined table of metrics for NumPy and PyTorch runs.

These results are the same as those reported in my OSPO presentation and Zenodo record on AniSOAP performance optimization.
