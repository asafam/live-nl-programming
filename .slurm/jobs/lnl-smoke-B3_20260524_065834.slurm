#!/bin/bash
#SBATCH --job-name=lnl-smoke-B3
#SBATCH --output=.slurm/logs/lnl-smoke-B3_%j.out
#SBATCH --error=.slurm/logs/lnl-smoke-B3_%j.err
#SBATCH --partition=cpu1T-24h

#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash /home/nlp/achimoa/workspace/live-nl-programming/scripts/run_pool_smoke_B.sh 
