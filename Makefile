#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

# Setup environment file.
env:
	cp -n .env.template .env

# Install rdkit
rdkit:
    conda install -c conda-forge -y rdkit

# Install GPU tensorflow
tensorflow-gpu:
	pip install tensorflow-gpu==1.15.0

# Install CPU tensorflow
tensorflow:
	pip install tensorflow==1.15.0


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".*.swp" -delete

## Lint using flake8
lint:
	flake8 src
