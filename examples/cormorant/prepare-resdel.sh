conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/residue_deletion

for MAXNUMAT in 500; do

	for NUMSHARDS in 5 10 15 20; do

		NPZDIR=../../data/residue_deletion/npz-maxnumat$MAXNUMAT-numshards$NUMSHARDS

		mkdir $NPZDIR
		mkdir $NPZDIR/resdel

		python ../../atom3d/datasets/res/convert_resdel_from_hdf5_to_npz.py $RAWDIR $NPZDIR/resdel --maxnumat $MAXNUMAT --numshards_tr $NUMSHARDS --numshards_va 3 --numshards_te 4

	done

done


