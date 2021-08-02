INFO_CSV="/oak/stanford/groups/rondror/projects/atom3d/supporting_files/ligand_efficacy_prediction/info.csv"
INPUT_FILES="/oak/stanford/groups/rondror/projects/atom3d/supporting_files/ligand_efficacy_prediction/pdb"
SPLIT_PATH="/oak/stanford/groups/rondror/projects/atom3d/ligand_efficacy_prediction/split/"
TEST=$SPLIT_PATH"test.csv"
TRAIN=$SPLIT_PATH"train.csv"
VAL=$SPLIT_PATH"val.csv"
OUTPUT_PATH="/oak/stanford/groups/rondror/projects/atom3d/ligand_efficacy_prediction/lmdb/"

python -m atom3d.datasets.lep.prepare_lmdb $INFO_CSV $INPUT_FILES $OUTPUT_PATH \
                                           -s -tr $TRAIN -va $VAL -te $TEST 

