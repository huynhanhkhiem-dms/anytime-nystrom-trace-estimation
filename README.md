# Anytime trace estimation from Gaussian Nyström progress

**Research reproducibility archive** for the manuscript of **Huynh Anh Khiem**
(Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam;
ORCID 0009-0007-7210-174X).

This repository preserves the numerical implementation, recorded results and
experiment protocols accompanying the manuscript submitted to the *SIAM Journal
on Matrix Analysis and Applications* (SIMAX). It is not an independently
validated software package.

## Contents

- `core.py`, `anchored_norm.py`, `progress_certificate.py`, `fast_evidence.py`,
  `baselines.py`: core implementations and certificates.
- `run_*.py` and calibration scripts: experiments; `validate_*.py` and
  `check_confirmatory_archive.py`: deterministic and archived-output checks.
- `results/`: archived experimental outputs. The public operator inputs are **not**
  redistributed; `data_sources.json` identifies the providers and SHA-256 checksums.
- `README.txt`: full commands and study-specific details.
- `manifest_sha256.json`: integrity hashes of the included research files.

## Reproduce the experiments

Use **Python 3.14** with the exact versions in `requirements.txt`. In the
repository root:

```bash
python -m pip install -r requirements.txt
python fetch_inputs.py
python run_anchor_norm_comparison.py --jobs 4
python run_na_checkpoint.py --jobs 4
python calibrate_na_checkpoint.py
python run_asymmetric_progress.py
python run_serial_block_timing.py
python run_verified_norm_comparison.py
python run_sparse_inverse.py
python analyze_results.py
```

See `README.txt` for the remaining experiment commands, order, numerical
assumptions, observed hardware/BLAS dependencies, and verification routines.
Theoretical guarantees assume exact Gaussian sampling and the conditions in
the manuscript; finite-precision and inexact solves require the stated verified
conditions. Random number seeds and operator provenance are recorded in the
code and supplementary experiment protocols.

To keep the archived results unchanged, rerun in a fresh checkout or copy.
The supplementary ZIP submitted with the paper is intended as a permanent
snapshot of these source and result files. File checksums are provided for
integrity checking; a matching checksum is not itself proof of mathematical
correctness.

## Citation and reuse

Please cite the associated manuscript if you use the work. Third-party input
data retain their original provider terms; check `data_sources.json` before
downloading or redistributing external data. No separate source-code license
is granted by this README.
