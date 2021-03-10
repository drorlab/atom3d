#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --gres gpu:1
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mvoegele@stanford.edu
#SBATCH --dependency=singleton
#SBATCH --job-name=msp-cutoff-CUTOFF-bs-BATCHSIZE-FORMAT-cormorant

module load gcc/8.1.0
module load cuda/10.0
source /home/users/mvoegele/miniconda3/etc/profile.d/conda.sh
conda activate cormorant

echo $CUDA_HOME

LMDBDIR=/oak/stanford/groups/rondror/projects/atom3d/lmdb/MSP/splits/split-by-sequence-identity-30/data
#LMDBDIR=/oak/stanford/groups/rondror/projects/atom3d/msp-corrected

python train.py --prefix msp_cutoff-CUTOFF_bs-BATCHSIZE_FORMAT --load \
                --datadir $FORMATDIR --format FORMAT\
                --ddir-suffix "_split-by-sequence-identity-30_cutoff-CUTOFF" \
                --radius CUTOFF \
                --batch-size BATCHSIZE \
                --num-epoch 50

