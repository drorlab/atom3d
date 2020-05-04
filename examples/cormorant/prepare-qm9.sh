conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/qm9/raw
NPZDIR=../../data/qm9/npz
IDXDIR=../../data/qm9/splits

mkdir $NPZDIR
mkdir $NPZDIR/qm9

python ../../atom3d/mpp/convert_qm9_from_sdf_to_npz.py $RAWDIR/gdb9.sdf.csv $RAWDIR/gdb9.sdf $NPZDIR/qm9 -i $IDXDIR

