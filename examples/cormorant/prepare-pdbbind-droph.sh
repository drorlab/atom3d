conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 


# -- Source directory paths --

RAWDIR=../../data/pdbbind/cnn3d
IDXDIR=../../data/pdbbind/cnn3d/splits/split_identity60


# -- Reduced datasets --

for MAXNUMAT in 500 300 320 340 360 380 400 420 440 460 480 520 540 560 580 600; do

	NPZDIR=../../data/pdbbind/npz_identity60_noh_maxnumat${MAXNUMAT}

	mkdir $NPZDIR
	mkdir $NPZDIR/pdbbind

	python ../../atom3d/datasets/lba/convert_pdbbind_from_hdf5_to_npz.py $RAWDIR $NPZDIR/pdbbind -i $IDXDIR --drop_h --maxnumat $MAXNUMAT

done


# -- The full dataset (still excluding rare-element structures) --

NPZDIR=../../data/pdbbind/npz_identity60_noh
mkdir $NPZDIR
mkdir $NPZDIR/pdbbind

python ../../atom3d/datasets/lba/convert_pdbbind_from_hdf5_to_npz.py $RAWDIR $NPZDIR/pdbbind -i $IDXDIR --drop_h


