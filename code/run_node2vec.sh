#!/bin/bash
echo "SUBMITTING NODE2VEC JOB"
sbatch -t 6-12 -o run_node2vec.log -N 1 --job-name="run_node2vec" --mem=32768 --wrap="python3 run_node2vec.py"