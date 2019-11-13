#!/bin/bash
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=10:30:30

source activate tf
./simphas/playmf.py --learning_rate 0.001 --n_episode 1000 --episode 400 --GAMMA 0.99 --h_speed $1 --s_speed $2 --opt $3 --seeds $4 --out $5 --vlag 0
source deactivate