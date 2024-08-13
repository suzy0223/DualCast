#!/bin/bash

function evalcmd () {

    echo $1

    eval $1

    sleep 0.5s

}



run_count=(1 2 3 4 5)
batch_size=32
max_epoch=100


for ((i=0; i<1; i++))
do
    wholecommand="python -u main.py --max_epoch ${max_epoch} --run_count ${run_count[$i]} --batch_size ${batch_size}"
    evalcmd "$wholecommand"
done
