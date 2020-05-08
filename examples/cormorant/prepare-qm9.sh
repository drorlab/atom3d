conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/qm9/raw
NPZDIR=../../data/qm9/npz
IDXDIR=../../data/qm9/splits

mkdir $NPZDIR
mkdir $NPZDIR/qm9
mkdir $IDXDIR

python ../../atom3d/mpp/convert_qm9_from_sdf_to_npz.py $RAWDIR $NPZDIR/qm9 -i $IDXDIR

