#!/bin/bash
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=99:99:99

source activate tf
./simphas/playmf.py --n_episode 500 --h_speed 3 --s_speed $1 --vlag 3 --out $2_$3_$4 --fileseeker policy_s_RMSprop_3_9.pkl --filehider policy_h_$5_3_$6.pkl --outflag 10
source deactivate
