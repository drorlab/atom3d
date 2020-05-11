#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	conda install -c conda-forge -y rdkit
	pip install -r requirements.txt

# Setup environment file.
env:
	cp -n .env.template .env

tensorflow-gpu:
	pip install tensorflow-gpu==1.15.0

tensorflow:
	pip install tensorflow==1.15.0
