# Reproduce every result in the repository.
#   make install     install the pinned package versions
#   make all         run both notebooks top to bottom (rewrites their outputs and every figure in results/)
#   make synthetic   synthetic study only (about 4 minutes)
#   make bosch       BOSCH study only: downloads the Zenodo data on the first run (about 7.4 GB),
#                    then builds the features and runs the experiments (about 15 minutes after the download)

PYTHON ?= python3
RUN = $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1

.PHONY: all install synthetic bosch

all: synthetic bosch

install:
	$(PYTHON) -m pip install -r requirements.txt

synthetic:
	$(RUN) Process_Study_Code_Only.ipynb

bosch:
	$(RUN) Real_Data_BOSCH_Experiment1.ipynb
