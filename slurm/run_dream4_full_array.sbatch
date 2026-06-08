#!/bin/bash
#SBATCH --job-name=grn_d4full_arr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-49%25
#SBATCH --output=grn_d4full_arr_%A_%a.out
#SBATCH --error=grn_d4full_arr_%A_%a.err

PYTHON=$HOME/.conda/envs/deeptgi/bin/python
cd /nas/longleaf/home/leyudai/GRN_benchmark

echo "=== DREAM4 Full-Data Array Task ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Python: $PYTHON"
echo ""

$PYTHON run_dream4_full_task.py --task-id "${SLURM_ARRAY_TASK_ID}"

echo ""
echo "End: $(date)"
