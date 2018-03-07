#!/bin/bash
module add python/3.5.1
echo "SUBMITTING NODE2VEC JOB WITH ARGS"
echo "$@"
p=$1
q=$2
COMMAND="python3 run_node2vec.py "
COMMAND+="$p"
COMMAND+=" "
COMMAND+="$q"
LOG="n2v_"
LOG+="$p"
LOG+="_"
LOG+="$q"
NAME="$LOG"
LOG+=".log"
sbatch -t 10-12 -o "$LOG" -N 1 --job-name="$NAME" --mem=100000 --wrap="$COMMAND"
echo "FINISHED"
