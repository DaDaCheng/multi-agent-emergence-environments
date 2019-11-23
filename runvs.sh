#!/bin/bash

for i in 1 4
do
        for j in SGLD RMSprop
        do
                for k in
                ./simphas/playmf.py $i $i --opt $j --seeds $4 --out $5 --vlag 0 --outflag 1


        done
done