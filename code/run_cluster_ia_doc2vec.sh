#!/bin/bash
echo "SUBMITTING CLUSTERING WITH ISSUEAREA JOB"
module add python/3.5.1
sbatch -t 1-12 -o ia_d2v_cluster.log -N 1 --job-name="ia_d2v_cluster" --mem=10000 --wrap="python3 cluster_ia_doc2vec.py"
echo "FINISHED CLUSTERING WITH ISSUEAREA."
