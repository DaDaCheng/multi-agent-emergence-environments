#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

source activate tf
# seeker
#./simphas/playmf.py --n_episode 1000 --h_speed $1 --s_speed 1 --out vs --vlag 3 --fileseeker $2.pkl --filehider $3.pkl



# Hider
./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed $1 --out vs --vlag 3 --fileseeker $2.pkl --filehider $3.pkl

