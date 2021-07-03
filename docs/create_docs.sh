#Installations needed:
#pip install sphinx
#pip install sphinx_rtd_theme
#pip install mock

# Remove old docs
rm -r source/* _build

# Auto-generate API docs:
sphinx-apidoc -M -o ./source ../atom3d \
	../atom3d/datasets/???/prepare*.py \
	../atom3d/datasets/???/process*.py \
	../atom3d/datasets/???/filter*.py \
	../atom3d/datasets/???/check*.py \
	../atom3d/datasets/???/gen*.py \
	../atom3d/datasets/???/get*.py \
	../atom3d/datasets/???/pyro*.py \
	../atom3d/datasets/???/bsa.py

# Format the auto-generated API docs
sed -i 's/ package//g' source/*.rst
sed -i 's/ module//g' source/*.rst
sed -i -e '/Submodules/,+2d' source/*.rst
sed -i -e '/Subpackages/,+2d' source/*.rst
#for FILE in source/*.rst; do
#       sed -i -n '/Module contents/q;p' $FILE
#done

# Build the docs from the .rst files
sphinx-build -b html . _build

