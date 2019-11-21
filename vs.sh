#!/bin/bash
for i in $1
do
        ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt SGLD --seeds $i --out SGLD_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001


        ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001
        #sbatch ./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed 1 --opt RMSprop --seeds $i --out RMSprop_speed1_$1 --vlag 0 --episode 400 --GAMMA 0.99 --learning_rate 0.001



done