#!/bin/bash
#SBATCH --job-name=lnl-smoke
#SBATCH --output=.slurm/logs/lnl-smoke_%j.out
#SBATCH --error=.slurm/logs/lnl-smoke_%j.err
#SBATCH --partition=cpu192G-48h

#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/run_pool_smoke.sh 
