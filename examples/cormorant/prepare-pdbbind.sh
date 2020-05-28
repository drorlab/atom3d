conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

MAXNUMAT=500

RAWDIR=../../data/pdbbind/cnn3d
NPZDIR=../../data/pdbbind/npz_identity60_maxnumat${MAXNUMAT}
IDXDIR=../../data/pdbbind/cnn3d/splits/split_identity60

mkdir $NPZDIR
mkdir $NPZDIR/pdbbind

python ../../atom3d/pli/convert_pdbbind_from_hdf5_to_npz.py $RAWDIR $NPZDIR/pdbbind -i $IDXDIR --maxnumat $MAXNUMAT


