#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=2:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate audioeval3
cd ~/data/audio_eval
python WinoSound/run_alm_on_conversations.py ${CHALLENGE_TYPE} ${PARA_CUE_TYPE} ${LAST_ONLY} ${ALM_NAME}

