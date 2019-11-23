#!/bin/bash
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=99:99:99

source activate tf
./simphas/playmf.py --h_speed $1 --s_speed $2 --opt $3 --seeds $4 --out $5 --vlag 0 --outflag 10 --n_episode 5000
./simphas/playmf.py --h_speed $1 --s_speed $2 --opt $3 --seeds $4 --out l$5 --vlag 0 --outflag 10 --n_episode 20000
source deactivate
