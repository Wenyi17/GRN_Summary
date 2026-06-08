# Result Tables

These TSV files contain the final benchmark results used in the manuscript.

## DREAM4

```text
dream4_full_results.tsv
```

DREAM4 full-data benchmark. Each row is one network, random seed, and method. The final table contains five DREAM4 networks x 10 random seeds x four methods.

```text
dream4_partitioned_results.tsv
```

DREAM4 partitioned-edge benchmark. The split is 60:20:20 training/validation/test edges, and methods are evaluated across 10 random seeds on held-out test pairs.

## BEELINE hESC

```text
bee_gp_cpu1.tsv
bee_gp_cpu2.tsv
bee_gp_cpu3.tsv
bee_gp_gpu.tsv
```

BEELINE-style balanced pair-level evaluation using 10-fold stratified cross-validation:

- `bee_gp_cpu1.tsv`: GENIE3
- `bee_gp_cpu2.tsv`: TIGRESS
- `bee_gp_cpu3.tsv`: iLSGRN
- `bee_gp_gpu.tsv`: DeepSEM, DeepTGI, GNNLink, and LINGER
