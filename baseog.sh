#!/bin/bash
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=99:99:99

source activate tf
./simphas/playmf.py --h_speed $1 --s_speed $2 --opt $3 --seeds $4 --out og$5_$6_$7 --vlag 0 --outflag 10 --n_episode 5000
./simphas/playmf.py --h_speed $1 --s_speed $2 --opt $3 --seeds $4 --out ogl$5_$6_$7 --vlag 0 --outflag 10 --n_episode 20000
source deactivate
