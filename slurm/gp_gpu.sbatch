#!/bin/bash
#SBATCH --job-name=gp_gpu
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load cuda/12.6

PYTHON=$HOME/.conda/envs/deeptgi/bin/python
cd /nas/longleaf/home/leyudai/GRN_benchmark

echo "=== GARNET protocol: DeepSEM + DeepTGI + GNNLink + LINGER (GPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"

$PYTHON -u run_beeline_garnet_protocol.py \
    --methods DeepSEM,DeepTGI,GNNLink,LINGER \
    --output results/bee_gp_gpu.tsv

echo "End: $(date)"
