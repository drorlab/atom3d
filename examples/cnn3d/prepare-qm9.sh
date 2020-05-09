conda activate /oak/stanford/groups/rondror/users/mvoegele/envs/atom3d 

RAWDIR=../../data/qm9/raw
HDFDIR=../../data/qm9/hdf5
IDXDIR=../../data/qm9/splits

mkdir $HDFDIR

python ../../atom3d/mpp/convert_qm9_from_sdf_to_hdf5.py $RAWDIR $HDFDIR -i $IDXDIR

