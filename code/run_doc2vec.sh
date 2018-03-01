#!/bin/bash
echo "SUBMITTING DOC2VEC JOB"
sbatch -t 6-12 -o run_doc2vec.log -N 1 --job-name="run_doc2vec" --mem=32768 --wrap="python3 run_doc2vec.py"
