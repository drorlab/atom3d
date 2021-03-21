#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --gres gpu:1
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mvoegele@stanford.edu
#SBATCH --dependency=singleton
#SBATCH --job-name=lba-id60-siamese-cutoff-CUTOFF-maxnumat-MAXNUM-cormorant

module load gcc/8.1.0
module load cuda/10.0
source /home/users/mvoegele/miniconda3/etc/profile.d/conda.sh
conda activate cormorant

echo $CUDA_HOME

LMDBDIR=/oak/stanford/groups/rondror/projects/atom3d/lmdb/LBA/splits/split-by-sequence-identity-60/data

python train.py --target neglog_aff --prefix lba-id60-siamese_cutoff-CUTOFF_maxnumat-MAXNUM --load \
                --datadir $LMDBDIR --format lmdb \
		--cgprod-bounded \
                --radius CUTOFF \
                --maxnum MAXNUM \
                --batch-size 1 \
                --num-epoch 150 \
                --siamese

