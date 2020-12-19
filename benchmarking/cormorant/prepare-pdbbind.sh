conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 


# -- Source directory paths --

RAWDIR=../../data/pdbbind/cnn3d
IDXDIR=../../data/pdbbind/cnn3d/splits/split_identity60


# -- The full dataset (still excluding rare-element structures) --

NPZDIR=../../data/pdbbind/npz_identity60
mkdir $NPZDIR
mkdir $NPZDIR/pdbbind

python ../../atom3d/pli/convert_pdbbind_from_hdf5_to_npz.py $RAWDIR $NPZDIR/pdbbind -i $IDXDIR


# -- Reduced datasets --

for MAXNUMAT in 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600; do

	NPZDIR=../../data/pdbbind/npz_identity60_maxnumat${MAXNUMAT}

	mkdir $NPZDIR
	mkdir $NPZDIR/pdbbind

	python ../../atom3d/pli/convert_pdbbind_from_hdf5_to_npz.py $RAWDIR $NPZDIR/pdbbind -i $IDXDIR --maxnumat $MAXNUMAT

done



