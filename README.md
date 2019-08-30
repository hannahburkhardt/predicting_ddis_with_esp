# Predicting DDIs with ESP

#### Repository author: Hannah Burkhardt (haalbu@uw.edu)

This repository contains the code used in our paper:

Burkhardt, Hannah A, Devika Subramanian, Justin Mower, and Trevor Cohen. 2019. “Predicting Adverse Drug-Drug Interactions with Neural Embedding of Semantic Predications.” To Appear in Proc AMIA Annu Symp 2019.

The corresponding data can be found on zenodo at https://zenodo.org/record/3333834/

Please also see the companion repository at https://github.com/hannahburkhardt/decagon, which contains the code necessary to run our implementation of the [Decagon algorithm](https://doi.org/10.1093/bioinformatics/bty294).

## Usage

Create a working directory. The notebook will create several folders in this directory.
```bash
mkdir predicting_ddis_with_esp
cd predicting_ddis_with_esp
```

Clone this git repository into a subfolder in your working directory.
```bash
git clone https://github.com/hannahburkhardt/predicting_ddis_with_esp.git
cd predicting_ddis_with_esp
```

Create a new conda environment with the given configuration, like so (depending on OS):
```bash
# MacOS
conda create -n predicting_ddis_with_esp_env --file jupyter_env_spec_file_osx.txt python=3.6.8
# Linux
conda create -n predicting_ddis_with_esp_env --file jupyter_env_spec_file_linux.txt python=3.6.8
```

Finally, run the Jupyter notebook:
```bash
conda activate predicting_ddis_with_esp_env
jupyter notebook .
```
