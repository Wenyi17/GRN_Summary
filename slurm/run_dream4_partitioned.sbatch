#!/bin/bash
#SBATCH --job-name=grn_d4part
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load cuda/12.6

PYTHON=$HOME/.conda/envs/deeptgi/bin/python
cd /nas/longleaf/home/leyudai/GRN_benchmark

echo "=== DREAM4 Partitioned Benchmark ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo ""

$PYTHON data/download_dream4.py
$PYTHON run_dream4_partitioned.py

echo ""
echo "End: $(date)"
