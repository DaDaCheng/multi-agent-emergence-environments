#!/bin/bash

for i in $2
do
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001


        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001




        simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 1 --out RMSprop_s_1_$i --vlag 3 --fileseeker policys_Admaspeed1.pkl --filehider policyh_RMSprop_speed1_$i.pkl --episode 1000
        #simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 2 --out RMSprop_s_2_$i --vlag 3 --fileseeker policys_RMSprop_speed1_$i.pkl --filehider policyh_Admaspeed2.pkl
        #simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 3 --out RMSprop_s_3_$i --vlag 3 --fileseeker policys_RMSprop_speed1_$i.pkl --filehider policyh_Admaspeed3.pkl
        #simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 4 --out RMSprop_s_4_$i --vlag 3 --fileseeker policys_RMSprop_speed1_$i.pkl --filehider policyh_Admaspeed4.pkl





        simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 1 --out SGLD_s_1_$i --vlag 3 --fileseeker policys_Admaspeed1.pkl --filehider policyh_SGLD_speed1_$i.pkl --episode 1000
        #simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 2 --out SGLD_s_2_$i --vlag 3 --fileseeker policys_SGLD_speed1_$i.pkl --filehider policyh_Admaspeed2.pkl
        #simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 3 --out SGLD_s_3_$i --vlag 3 --fileseeker policys_SGLD_speed1_$i.pkl --filehider policyh_Admaspeed3.pkl
        #simphas/playmf.py --n_episode $1 --h_speed 1 --s_speed 4 --out SGLD_s_4_$i --vlag 3 --fileseeker policys_SGLD_speed1_$i.pkl --filehider policyh_Admaspeed4.pkl







done