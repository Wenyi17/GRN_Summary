#!/bin/bash
#SBATCH --job-name=grn_d4full_merge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=grn_d4full_merge_%j.out
#SBATCH --error=grn_d4full_merge_%j.err

PYTHON=$HOME/.conda/envs/deeptgi/bin/python
cd /nas/longleaf/home/leyudai/GRN_benchmark

echo "=== Merge DREAM4 Full-Data Results ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

$PYTHON combine_dream4_full_results.py

echo ""
echo "End: $(date)"
