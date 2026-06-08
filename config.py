"""
Central configuration for the GRN benchmark pipeline.
All paths, hyperparameters, and method lists live here.
"""
import os
from pathlib import Path


def _path_from_env(name, default):
    """Return a pathlib.Path from an environment variable or default path."""
    return Path(os.environ.get(name, default)).expanduser().resolve()


# ──────────────────────────────────────────────────────────
# DIRECTORIES
# ──────────────────────────────────────────────────────────
ROOT         = _path_from_env("GRN_BENCHMARK_ROOT", Path(__file__).resolve().parent)
DATA_DIR     = ROOT / "data"
DREAM4_DIR   = DATA_DIR / "DREAM4"
BEELINE_DIR  = _path_from_env("BEELINE_DIR", ROOT / "external_data" / "Beeline")
RESULTS_DIR  = ROOT / "results"
PLOTS_DIR    = ROOT / "plots"

# ──────────────────────────────────────────────────────────
# DREAM4 SETTINGS
# ──────────────────────────────────────────────────────────
DREAM4_NETS      = [1, 2, 3, 4, 5]
DREAM4_N_GENES   = 100

# ──────────────────────────────────────────────────────────
# DREAM4 PARTITIONED SETTINGS
# ──────────────────────────────────────────────────────────
PARTITION_SEED   = 42
PARTITION_TRAIN  = 0.6
PARTITION_VAL    = 0.2
PARTITION_TEST   = 0.2

# ──────────────────────────────────────────────────────────
# BEELINE hESC SETTINGS
# ──────────────────────────────────────────────────────────
BEELINE_EXPR  = _path_from_env("BEELINE_EXPR", BEELINE_DIR / "BEELINE-data/inputs/scRNA-Seq/hESC/ExpressionData.csv")
BEELINE_PT    = _path_from_env("BEELINE_PT", BEELINE_DIR / "BEELINE-data/inputs/scRNA-Seq/hESC/PseudoTime.csv")
BEELINE_GOLD  = _path_from_env("BEELINE_GOLD", BEELINE_DIR / "Networks/human/hESC-ChIP-seq-network.csv")
BEELINE_TFS   = _path_from_env("BEELINE_TFS", BEELINE_DIR / "human-tfs.csv")

# ──────────────────────────────────────────────────────────
# METHOD LISTS PER EXPERIMENT
# ──────────────────────────────────────────────────────────
DREAM4_FULL_METHODS        = ["GENIE3", "GRNBoost2", "iLSGRN", "TIGRESS"]
DREAM4_PARTITIONED_METHODS = ["LINGER", "GNNLink", "DeepSEM", "DeepTGI"]
BEELINE_METHODS            = (DREAM4_FULL_METHODS + DREAM4_PARTITIONED_METHODS)

# ──────────────────────────────────────────────────────────
# TRAINING HYPER-PARAMETERS (deep-learning methods)
# ──────────────────────────────────────────────────────────
SEED           = 0
BATCH_SIZE     = 256
EPOCHS_DEEPSEM = 300
EPOCHS_DEEPTGI = 40
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
K_FOLDS        = 5
HIDDEN_DIM     = 256
