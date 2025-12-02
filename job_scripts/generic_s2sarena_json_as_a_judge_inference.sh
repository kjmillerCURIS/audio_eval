#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=5:59:59
#$ -l gpus=1
#$ -l gpu_c=8.6
#$ -pe omp 7
#$ -j y
#$ -m ea

module load miniconda
conda activate audioeval3
cd ~/data/audio_eval
python json_as_a_judge/s2sarena_json_as_a_judge_inference.py ${PARAMS_KEY} ${REP}

