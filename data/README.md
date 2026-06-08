# Data Files

This repository does not include large raw external datasets.

## Included

The study-specific DREAM4 partitioned benchmark inputs are included:

```text
DREAM4/partitioned/all_expression.tsv
DREAM4/partitioned/train_edges.tsv
DREAM4/partitioned/val_edges.tsv
DREAM4/partitioned/test_edges.tsv
```

These files define the 60:20:20 train/validation/test edge split used by `run_dream4_partitioned.py`.

## Not Included

Raw DREAM4 full-data expression and gold-standard files are not included. Use `download_dream4.py` to prepare them locally from the original DREAM4 download.

BEELINE hESC raw input files are not included. Set `BEELINE_DIR`, or set `BEELINE_EXPR` and `BEELINE_GOLD` directly before running the BEELINE benchmark.
