#!/bin/bash


for i in 1 2 3 4
do
        for j in SGDL RMSprop
        do
                for k in {1..25}
                do
                        sbatch ./vs.sh $i $k$j$i $i $j$k
                #sleep 1
                done
        done


done
#policys_base1RMSprop10.pkl
#./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed $1 --out $2vs --vlag 3 --fileseeker policys_base$3Adma6.pkl --filehider policyh_base1$4.pkl