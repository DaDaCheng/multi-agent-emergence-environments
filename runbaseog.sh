#!/bin/bash


for i in 1 2 3 4
do
  for j in SGLD RMSprop
  do
    for k in {1..20}
    do
      sbatch baseog.sh $i $i $j $k $j $i $k
    done

  done
done


for i in 1 2 3 4
do
  for j in Adma
  do
    for k in 1 2 3 4
    do
     sbatch baseog.sh $i $i $j $k $j $i $k
    done

  done
done
