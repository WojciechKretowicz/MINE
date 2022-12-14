#!/bin/bash
#SBATCH --job-name=mine_n_rem # Job name	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=64                  # Number of processes
#SBATCH --output=multiprocess_%j.log # Standard output and error log
date;hostname;pwd

python3 neighbours_task_remaining.py

date
