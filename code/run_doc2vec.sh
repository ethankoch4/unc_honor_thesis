#!/bin/bash
echo "SUBMITTING DOC2VEC JOB"
sbatch -t 2-2 -o run_doc2vec.log --job-name="run_doc2vec" --wrap="python3 run_doc2vec.py"
