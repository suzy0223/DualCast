#!/bin/bash

function evalcmd () {

    echo $1

    eval $1

    sleep 0.5s

}


run_count=(1 2 3 4 5)
max_epoch=100
batch_size=16


for ((i=0; i<1; i++))
do
    wholecommand="python -u main.py --batch_size ${batch_size} --max_epoch ${max_epoch} --run_count ${run_count[$i]}"
    evalcmd "$wholecommand"
done
