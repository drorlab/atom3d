conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/residue_deletion

MAXNUMAT=500

for NUMSHARDS in 10 20 30 40 50 60 80 100 140 200; do

	NPZDIR=../../data/residue_deletion/npz-maxnumat$MAXNUMAT-numshards$NUMSHARDS

	mkdir $NPZDIR
	mkdir $NPZDIR/resdel

	python ../../atom3d/residue_deletion/convert_resdel_from_hdf5_to_npz.py $RAWDIR $NPZDIR/resdel --maxnumat $MAXNUMAT --numshards $NUMSHARDS

done


