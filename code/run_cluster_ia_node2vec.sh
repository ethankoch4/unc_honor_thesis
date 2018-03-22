#!/bin/bash
echo "SUBMITTING CLUSTERING WITH ISSUEAREA JOB"
module add python/3.5.1
sbatch -t 4-12 -o ia_n2v_cluster.log -N 1 --job-name="ia_n2v_cluster" --mem=99000 --wrap="python3 cluster_ia_node2vec.py 1.0 1.0"
echo "FINISHED CLUSTERING WITH ISSUEAREA."
