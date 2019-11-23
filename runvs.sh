#!/bin/bash

for i in 1 2 3 4
do
for j in 1 3 6 9 11 12 17 20
do

	sbatch ./vs.sh $i SGLD $i $j SGLD $j

done
done





for i in 1 2 3 4
do
for j in 4 6 8 9
do

        sbatch ./vs.sh $i RMSprop $i $j RMSprop $j

done
done
