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
simphas/playmf.py --n_episode 200 --h_speed $1 --s_speed 4 --out $2vh --vlag 3   --filehider policyh_base$3Adma5.pkl  --fileseeker policys_base4$4$5.pkl


# Hider
#simphas/playmf.py --n_episode 200 --h_speed 4 --s_speed $1 --out $2vs --vlag 3 --fileseeker policys_base$3Adma5.pkl --filehider policyh_base4$4$5.pkl
source deactivate