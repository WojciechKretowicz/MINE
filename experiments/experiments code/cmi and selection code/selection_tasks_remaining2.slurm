#!/bin/bash
#SBATCH --job-name=mine_neighbours # Job name	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=70                   # Number of processes
#SBATCH --output=multiprocess_%j.log # Standard output and error log
date;hostname;pwd

python3 selection_tasks_remaining2.py

date
