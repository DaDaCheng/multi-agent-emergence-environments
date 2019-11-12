#!/bin/bash


for i in 1 2 3 4
do
  for j in SGDL RMSprop
  do
    for k in {1..25}
    do
      #sbatch ./vs.sh $i $i $j $k base$i$j$k

    done
  done


done

#./simphas/playmf.py --n_episode 1000 --h_speed 1 --s_speed $1 --out vs --vlag 3 --fileseeker $2.pkl --filehider $3.pkl
