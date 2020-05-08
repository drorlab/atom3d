#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	conda install -c conda-forge -y rdkit
	pip install -r requirements.txt

tensorflow-gpu:
	pip install tensorflow-gpu==1.15.0

tensorflow:
	pip install tensorflow==1.15.0
