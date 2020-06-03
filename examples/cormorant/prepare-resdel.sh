conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/residue_deletion

for MAXNUMAT in 320 500; do

	for NUMSHARDS in 10 10; do

		NPZDIR=../../data/residue_deletion/npz-maxnumat$MAXNUMAT-numshards$NUMSHARDS

		mkdir $NPZDIR
		mkdir $NPZDIR/resdel

		python ../../atom3d/residue_deletion/convert_resdel_from_hdf5_to_npz.py $RAWDIR $NPZDIR/resdel --maxnumat $MAXNUMAT --numshards_tr $NUMSHARDS --numshards_va 3 --numshards_te 4

	done

done


