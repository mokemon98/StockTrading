#!/bin/sh
array=("1" "2" "3" "4" "5")
for dir in "${array[@]}"
do
    #echo $dir
    python stock_classification.py $1 $dir --gpu 1 &
done
