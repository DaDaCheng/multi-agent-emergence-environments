#!/bin/bash


for i in 1 2 3 4
do
  for j in SGLD RMSprop
  do
    for k in {1..12}
    do
      sbatch basepolicy.sh $i $i $j $k $j_s$i_$k
    done

  done
done


for i in 1 2 3 4
do
  for j in Adma
  do
    for k in 1 2
    do
     sbatch basepolicy.sh $i $i $j $k $j_s$i_$k
    done

  done
done
