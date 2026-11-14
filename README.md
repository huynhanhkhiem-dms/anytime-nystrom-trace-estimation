# Anytime trace estimation from Gaussian Nyström progress

Reproducibility code and recorded numerical results for the manuscript by **Huynh Anh Khiem**.

## Setup

Use Python 3.14 in a new virtual environment. Run the commands from the repository root.

```sh
python -m venv .venv
# On macOS or Linux
. .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_inputs.py
```

The input script downloads 12 files from the public providers listed in [`data_sources.json`](data_sources.json) and checks their SHA-256 digests. The sources’ terms apply to those inputs.

## Reproduce the results

Use [`README.txt`](README.txt) for the experiment commands, run order, and validation checks. From a fresh checkout, `python analyze_results.py` regenerates both figures from the recorded outputs in [`results/`](results/). The supplied outputs can be compared with a rerun in a separate working copy. Runtime and iterative-solver steps can vary with the host and BLAS.

[`CONTENTS.txt`](CONTENTS.txt) lists the files in the accompanying journal archive. [`manifest_sha256.json`](manifest_sha256.json) records their checksums, apart from the manifest itself. [`source_provenance.json`](source_provenance.json) maps source hashes from the experiment records to the supplied files.
