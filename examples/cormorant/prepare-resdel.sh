conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/residue_identity/environments-split

for MAXNUMAT in 500; do

	for NUMSHARDS in 20 20 25 25; do

		NPZDIR=../../data/residue_identity/environments-split/npz-maxnumat$MAXNUMAT-numshards$NUMSHARDS

		mkdir $NPZDIR
		mkdir $NPZDIR/resdel

		python ../../atom3d/datasets/res/convert_resdel_from_hdf5_to_npz.py $RAWDIR $NPZDIR/resdel --maxnumat $MAXNUMAT --numshards_tr $NUMSHARDS --numshards_va 3 --numshards_te 4

	done

done


