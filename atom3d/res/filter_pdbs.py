import os
import shutil

from tqdm import tqdm

train = set()
with open('/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion/train.txt') as f:
    for line in f:
        train.add(line.strip()) 
val = set()
with open('/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion/val.txt') as f:
    for line in f:
        val.add(line.strip()) 
test = set()
with open('/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion/test.txt') as f:
    for line in f:
        test.add(line.strip()) 

src_dir = '/oak/stanford/groups/rbaltman/aderry/graph-pdb/data/raw'
dest_dir = '/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion'
for f in tqdm(os.listdir(src_dir)):
    if f[:4] in train:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, 'train'))
    elif f[:4] in val:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, 'val'))
    elif f[:4] in test:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, 'test'))


