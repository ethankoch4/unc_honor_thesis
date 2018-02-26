#!/bin/bash
echo "SUBMITTING NODE2VEC JOB WITH ARGS"
echo "$@"
p=$1
q=$2
COMMAND="python3 run_node2vec.py "
COMMAND+="$p"
COMMAND+=" "
COMMAND+="$q"
LOG="n2v_run_"
LOG+="$p"
LOG+="_"
LOG+="$q"
NAME="$LOG"
LOG+=".log"
sbatch -t 6-12 -o "$LOG" -N 1 --job-name="$NAME" --mem=49152 --wrap="$COMMAND"
echo "FINISHED"
