Anytime trace estimation from Gaussian Nyström progress
Huynh Anh Khiem (ORCID 0009-0007-7210-174X)

Setup
-----
Extract this archive into a writable directory. Use Python 3.14 and the
package versions in requirements.txt. Download and verify the 12 inputs
from the public data providers listed in data_sources.json.

    python -m pip install -r requirements.txt
    python fetch_inputs.py

Primary experiments
-------------------
Run these commands from the extracted directory after fetching the inputs:

    python run_anchor_norm_comparison.py --jobs 4
    python run_na_checkpoint.py --jobs 4
    python calibrate_na_checkpoint.py
    python run_asymmetric_progress.py
    python run_serial_block_timing.py
    python run_verified_norm_comparison.py
    python run_sparse_inverse.py

The original NA-Hutch++ checkpoint experiment writes to the working
directory, where calibrate_na_checkpoint.py reads it. The results/
directory contains the recorded experimental outputs used in the paper.
Rerun on a copy of this archive to keep those results available for
comparison. Each driver documents its seeds. The operator inputs and SHA-256
digests are recorded in data_sources.json. Source terms apply to the inputs.

Additional experiments
----------------------
The following commands reproduce analyses described in the main paper
or Supplement.pdf. Run the pilot and structural commands in the order shown.

    python run_pilot.py --seeds 100 --budget 55 --name pilot_expanded
    python compare_pilot.py
    python chi2_comparator.py --suite all
    python run_progress_pilot.py --matrices bcsstk14 --name structural_progress_pilot
    python run_progress_pilot.py --matrices bcsstk15 bcsstk16 --name structural_progress_remainder
    python run_split_checkpoints.py
    python run_graph_holdout.py
    python run_confidence_sweep.py
    python adaptive_hutch_baseline.py
    python run_timing_check.py
    python run_confirmatory.py --jobs 4
    python run_stake_sensitivity.py --jobs 3
    python run_verified_galerkin.py
    python run_inexact_galerkin.py
    python run_stake_ablation.py --jobs 4
    python run_same_risk_replay.py
    python run_controlled_timing.py
    python run_single_report_comparison.py

Verification and figures
------------------------

    python analyze_results.py
    python validate_progress.py
    python validate_numeric_extremes.py
    python validate_anchored_norm.py
    python validate_fast_evidence.py
    python validate_baselines.py
    python validate_inexact_endpoints.py
    python validate_inexact_galerkin.py
    python validate_full_span.py
    python validate_anchor_norm_comparison.py
    python check_confirmatory_archive.py
    python validate_archived_replays.py

The figure script regenerates both figures from the recorded results and
writes paired_analysis_recomputed.json. The result files retain the original
source digests where applicable. source_provenance.json maps source names
and hashes from the original records to the delivered files. Some records
use a binary64 variant of the progress calculations for ordinary scales.
The validation scripts recognize its recorded digest. manifest_sha256.json
records the exact bytes in this archive. The names of some output files and
metadata fields were updated for this package. The numerical records were
retained.

The theorem uses exact Gaussian draws and exact products. The inverse
implementation uses outward arithmetic and a certified lower spectral
bound. Wall times and the inner solve effort depend on the host and BLAS.
In the sparse benchmark rerun reported in the supplement, 11 of 12 probe
counts matched. A single progress run used 120 in place of 119 products.
Both operator medians and all containment outcomes matched.

