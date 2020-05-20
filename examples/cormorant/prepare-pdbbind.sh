conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/pdbbind/cnn3d
NPZDIR=../../data/pdbbind/npz
IDXDIR=../../data/pdbbind/cnn3d/splits

mkdir $NPZDIR
mkdir $NPZDIR/pdbbind

python ../../atom3d/pli/convert_pdbbind_from_hdf5_to_npz.py $RAWDIR $NPZDIR/pdbbind -i $IDXDIR

