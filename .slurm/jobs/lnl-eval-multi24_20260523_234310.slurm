#!/bin/bash
#SBATCH --job-name=lnl-eval-multi24
#SBATCH --output=.slurm/logs/lnl-eval-multi24_%j.out
#SBATCH --error=.slurm/logs/lnl-eval-multi24_%j.err
#SBATCH --partition=cpu192G-48h

#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=48
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/run_pool_eval.sh 
