#!/bin/bash
#SBATCH --job-name=lnl-smoke-B6
#SBATCH --output=.slurm/logs/lnl-smoke-B6_%j.out
#SBATCH --error=.slurm/logs/lnl-smoke-B6_%j.err
#SBATCH --partition=cpu192G-48h

#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/run_pool_smoke_B.sh 
