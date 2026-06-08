#!/bin/bash
#SBATCH --job-name=grn_d4full
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

echo "=== DREAM4 Full-Data Benchmark ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

$PYTHON data/download_dream4.py
$PYTHON run_dream4_full.py

echo ""
echo "End: $(date)"
