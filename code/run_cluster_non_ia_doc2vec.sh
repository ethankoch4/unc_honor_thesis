#!/bin/bash
echo "RUNNING cluster non issueArea PYTHON SCRIPT STARTING WITH N_CLUSTERS:"
var=$1
echo "$1"
echo "AND ENDING AT:"
finish=$2
echo "$2"
echo "WITH STEP SIZE:"
step=$3
echo "$3"
while [ "$var" -gt "$finish" ]
do
    echo "$var"
    COMMAND="python3 cluster_non_ia_doc2vec.py "
    COMMAND+="$var"
    echo "$COMMAND"
    OUT_NAME="non_ia_d2v_"
    OUT_NAME+="$var"
    OUT_NAME+=".log"
    JOB_NAME="non_ia_d2v"
    JOB_NAME+="$var"
    sbatch -o "$OUT_NAME" -t 3-12 --job-name="$JOB_NAME" --mem=16384 --wrap="$COMMAND"
    var=$(($var-$step))
done
echo "COMPLETED cluster non issueArea PYTHON SCRIPT"
