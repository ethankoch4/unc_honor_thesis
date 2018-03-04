#!/bin/bash
echo "SUBMITTING CLUSTERING WITH ISSUEAREA JOB"
sbatch -t 3-12 -o ia_d2v_cluster.log -N 1 --job-name="ia_d2v_cluster" --mem=32768 --wrap="python3 cluster_ia_doc2vec.py"
echo "FINISHED CLUSTERING WITH ISSUEAREA."
