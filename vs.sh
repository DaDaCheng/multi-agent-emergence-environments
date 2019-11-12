#!/bin/bash
#
#
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=30:30:30

source activate tf
# seeker
#./simphas/playmf.py --n_episode 1000 --h_speed $1 --s_speed 1 --out vs --vlag 3 --fileseeker $2.pkl --filehider $3.pkl



# Hider
/simphas/playmf.py --n_episode 200 --h_speed 1 --s_speed $1 --out $2vs --vlag 3 --fileseeker policys_base$3Adma6.pkl --filehider policyh_base1$4.pkl
source deactivate
