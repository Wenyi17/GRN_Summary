#!/bin/bash
#SBATCH --job-name=gp_cpu3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

PYTHON=$HOME/.conda/envs/deeptgi/bin/python
cd /nas/longleaf/home/leyudai/GRN_benchmark

echo "=== GARNET protocol: iLSGRN (CPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

$PYTHON -u run_beeline_garnet_protocol.py \
    --methods iLSGRN \
    --output results/bee_gp_cpu3.tsv

echo "End: $(date)"
