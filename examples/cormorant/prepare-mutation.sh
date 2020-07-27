conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

#RAWDIR=../../data/mutation_prediction
RAWDIR=../../../../../projects/atom3d/mutation_stability_prediction

for CUTOFF in 05 06 07 08 09 10 11 12 14 16 20 24 30; do

	NPZDIR=../../data/mutation_prediction/npz-cutoff$CUTOFF

	mkdir $NPZDIR
	mkdir $NPZDIR/mutation

	python ../../atom3d/datasets/msp/convert_mutation_from_hdf5_to_npz.py $RAWDIR $NPZDIR/mutation --cutoff $CUTOFF 

done


