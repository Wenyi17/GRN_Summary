#!/bin/bash
#SBATCH --job-name=gp_cpu1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

PYTHON=$HOME/.conda/envs/deeptgi/bin/python
cd /nas/longleaf/home/leyudai/GRN_benchmark

echo "=== GARNET protocol: GENIE3 + GRNBoost2 (CPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

$PYTHON -u run_beeline_garnet_protocol.py \
    --methods GENIE3,GRNBoost2 \
    --output results/bee_gp_cpu1.tsv

echo "End: $(date)"
