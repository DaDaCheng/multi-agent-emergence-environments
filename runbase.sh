#!/bin/bash


for i in 1
do
  for j in SGDL RMSprop
  do
    for k in {1..25}
    do
      sbatch ./basepolicy.sh $i $i $j $k base$i$j$k

    done
  done


done

for i in 1 2 3 4
do
  for j in Adma
  do
    for k in {5..8}
    do
      sbatch ./basepolicy.sh $i $i $j $k base$i$j$k

    done
  done


done