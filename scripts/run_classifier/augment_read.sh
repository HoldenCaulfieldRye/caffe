#!/bin/bash
numer=$(cat train.txt | grep '0$' | wc -l);
denom=$(cat train.txt | wc -l);
thresh=$(echo "scale=2; $numer/$denom" | bc);
if [ $(echo " $thresh < 0.5" | bc) -eq 1 ];
then
    echo "0$thresh threshold" >> read.txt
    echo "0 flag_val" >> read.txt
else
    thresh=$(echo "1 - $thresh" | bc)
    echo "0$thresh threshold" >> read.txt
    echo "1 flag_val" >> read.txt
fi
