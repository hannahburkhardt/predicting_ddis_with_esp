# Predicting DDIs with ESP

#### Repository author: Hannah Burkhardt (haalbu@uw.edu)

This repository contains the code used in our paper:

Burkhardt, Hannah A, Devika Subramanian, Justin Mower, and Trevor Cohen. 2019. “Predicting Adverse Drug-Drug Interactions with Neural Embedding of Semantic Predications.” To Appear in Proc AMIA Annu Symp 2019.

The corresponding data can be found on zenodo at [TBD]

Please also see the companion repository at https://github.com/hannahburkhardt/decagon, which contains the code necessary to run out implementation of the [Decagon algorithm](https://doi.org/10.1093/bioinformatics/bty294).

## Usage

```bash
git clone https://github.com/hannahburkhardt/predicting_ddis_with_esp.git
cd predicting_ddis_with_esp
```

After cloning the repository, create a new conda environment with the given configuration like so:
```bash
conda create -n decagon_jupyter --file jupyter_env_spec_file.txt python=3.6.8
```

Finally, run the Jupyter notebook:
```bash
conda activate decagon_jupyter
jupyter notebook .
```
