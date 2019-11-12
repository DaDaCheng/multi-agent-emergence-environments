#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

source activate tf
./simphas/playmf.py --n_episode 1000 --h_speed $1 --s_speed $2 --opt $3 --seeds $4 --out $5 --vlag 0