# GRN Benchmark Code and Results

This repository contains the benchmark/training code and result tables used for the GRN inference review manuscript, **A Data-Centric Evidence-Aware Review of Gene Regulatory Network Inference**.

The repository intentionally excludes figure-generation code and large raw external datasets. It keeps the benchmark scripts, method wrappers, evaluation utilities, SLURM training scripts, final result tables, and the DREAM4 partitioned split that was generated for this study.

## Repository Layout

- `config.py`: central paths and benchmark settings.
- `run_dream4_full.py`: sequential DREAM4 full-data benchmark.
- `run_dream4_full_task.py`: one DREAM4 full-data network-seed task for array jobs.
- `combine_dream4_full_results.py`: merge DREAM4 full-data array outputs.
- `run_dream4_partitioned.py`: DREAM4 60:20:20 edge-split benchmark.
- `run_beeline_garnet_protocol.py`: BEELINE hESC balanced pair protocol.
- `data/`: DREAM4 preparation scripts and study-specific partitioned inputs.
- `evaluation/`: AUROC, AUPRC, and early precision metrics.
- `methods/`: method wrappers used in the benchmarks.
- `results/`: final benchmark result tables.
- `slurm/`: SLURM scripts for running benchmarks.

## Benchmark Protocols

### DREAM4 Full-Data

`run_dream4_full.py` evaluates GENIE3, GRNBoost2, iLSGRN, and TIGRESS on the five DREAM4 InSilico Size100 Multifactorial networks. There is no train/validation/test split in this setting. Each method infers a complete network from each full expression matrix, and predictions are scored against the corresponding network-specific gold standard.

The current result table uses 10 random seeds across five DREAM4 networks, giving 50 evaluation points per method.

For faster HPC execution:

```bash
sbatch slurm/run_dream4_full_array.sbatch
sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/combine_dream4_full.sbatch
```

Output:

```text
results/dream4_full_results.tsv
```

### DREAM4 Partitioned

`run_dream4_partitioned.py` evaluates LINGER, GNNLink, DeepSEM, and DeepTGI under a supervised/topology-aware edge-prediction setting. Gold-standard TF-target pairs are split into non-overlapping 60:20:20 training, validation, and test edge sets. This partitioned DREAM4 benchmark is study-specific and is retained under:

```text
data/DREAM4/partitioned/
```

Output:

```text
results/dream4_partitioned_results.tsv
```

### BEELINE hESC

`run_beeline_garnet_protocol.py` implements the BEELINE-style hESC evaluation used in the manuscript. It constructs balanced positive/negative TF-target pairs and evaluates methods with 10-fold stratified pair-level cross-validation. This setting is not a 60:20:20 train/validation/test split and does not use a separate validation set.

Outputs:

```text
results/bee_gp_cpu1.tsv   # GENIE3
results/bee_gp_cpu2.tsv   # TIGRESS
results/bee_gp_cpu3.tsv   # iLSGRN
results/bee_gp_gpu.tsv    # DeepSEM, DeepTGI, GNNLink, LINGER
```

## Data Notes

Raw DREAM4 expression and gold-standard files are not included. To rerun the DREAM4 full-data benchmark, place the original DREAM4 files locally and run:

```bash
python data/download_dream4.py
```

The study-specific DREAM4 partitioned inputs are included because they define the 60:20:20 edge split used in the manuscript.

BEELINE hESC raw input files are not included. Set one of the following before running the BEELINE benchmark:

```bash
export BEELINE_DIR=/path/to/Beeline
```

or set the files directly:

```bash
export BEELINE_EXPR=/path/to/ExpressionData.csv
export BEELINE_GOLD=/path/to/hESC-ChIP-seq-network.csv
```

## Environment

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

The GPU-enabled methods require a PyTorch installation compatible with the local CUDA environment. On HPC, edit the `PYTHON` variable in `slurm/*.sbatch` if your environment uses a different Python path.

## Running Benchmarks

```bash
python run_dream4_full.py
python run_dream4_partitioned.py
python run_beeline_garnet_protocol.py --methods GENIE3,GRNBoost2 --output results/bee_gp_cpu1.tsv
python run_beeline_garnet_protocol.py --methods TIGRESS --output results/bee_gp_cpu2.tsv
python run_beeline_garnet_protocol.py --methods iLSGRN --output results/bee_gp_cpu3.tsv
python run_beeline_garnet_protocol.py --methods DeepSEM,DeepTGI,GNNLink,LINGER --output results/bee_gp_gpu.tsv
```

## Notes

The method wrappers are intended for manuscript-level benchmarking and protocol comparison. They adapt or reimplement published methods in a common evaluation pipeline; for production use of individual methods, consult the original method repositories and documentation.
