# AF-guided MR
A Python tool that automates molecular replacement using protein sequences and x-ray diffraction data, designed especially to handle difficult cases.

This tool uses protein sequences and reduced x-ray diffraction data to automate the process of molecular replacement. It leverages the power of ColabFold for initial structure prediction and Phaser for MR based on various predefined modes. For high-resolution cases better than 3.5 angstroms, it uses AutoBuild to enhance and build the model. For low-resolution cases worse than 3.5 angstroms, it uses phenix.refine to run default refinement cycles for a brief and quick assessment of the molecular replacement correctness. It is specifically designed to handle difficult cases where the predicted structure varies significantly from the final solution.

## Getting Started

The tool consists of a series of Python scripts:
1. `main.py` (the main script)
2. `AF_cluster.py` (used for handling special cases)

### Prerequisites

You'll need to have the following installed on a Linux machine (preferably Ubuntu) with Nvidia GPU to run the scripts:
- conda (or mamba)
- Python 3.10 (preferably within a Conda environment)
- PHENIX (Python-based Hierarchical ENvironment for Integrated Xtallography) (https://www.phenix-online.org/)

### Installation
#### Install PHENIX
You can follow the [official PHENIX installation guide](https://www.phenix-online.org/download/) for detailed instructions. The installation process is straightforward and should be completed within a few minutes.

#### Install Nvidia GPU drivers with CUDA support
You can follow the [official Nvidia installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for detailed instructions, if this is the first time you are installing Nvidia GPU drivers with CUDA support.
#### Install Conda
You can follow the [official Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for detailed instructions.
[Optionally, you can install and use mamba for faster environment management: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html. This is not necessary and will require some additional configuration but can be helpful for faster environment management.]

After installing Conda, create a new Conda environment with Python 3.10:
```
conda create -n automatemr python=3.10
# mamba create -n automatemr python=3.10 if you use mamba, similar to following commands, details: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
```
you can choose other names for your environment.
Activate your newly created Conda environment:
```
conda activate automatemr
```
#### install ColabFold
Although local-colabfold should work as well, I would recommend the default ColabFold, as this python function was tested with the default version. The following installation process consulted the ColabFold installation guide and local-colabfold installation guide. 
```
conda install -c conda-forge -c bioconda openmm==7.7.0 pdbfixer kalign2=2.04 hhsuite=3.3.0 cctbx-base 
# cctbx-base can be installed later but doesn't hurt to install it now
pip install --no-warn-conflicts "colabfold[alphafold-without-jax] @ git+https://github.com/sokrypton/ColabFold"
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "colabfold[alphafold]"
```
When you use pip, make sure you are in the right conda environment, otherwise you are just using the system python and pip, not the conda environment you just created. You can check your pip and python version by `which pip` and `which python`. 

One word on Ada Lovelace cards: if you use e.g. RTX 4080, 4090 or ADA6000 cards, you will need to upgrade CUDA to 11.8 or later so that your Ada Lovelace cards can use CUDA. Then your python will need to be at least 3.9 and jax with the right CUDA support. Above installation process should work for Ada Lovelace cards but if you run into problems, check your CUDA version and python version.

If you encounter problem when running colabfold because of pdbfixer, like `ModuleNotFoundError: No module named 'simtk.openmm.app.internal'`, then you may need to do the following:
use a text editor, like nano or vim, open your pdbfixer.py, e.g.
`nano ~/anaconda3/envs/automatemr/lib/python3.9/site-packages/pdbfixer/pdbfixer.py`
then replace every instance of `simtk.openmm` with just `openmm`.

## Usage
First, clone the repository and optionally add the main script to your bash_aliases for easy access. Then, install the required packages and you are ready to go. For example, you can do the following in your terminal to install and use the main script:
```
git clone https://github.com/ww2283/AF-guided-MR
nano ~/.bash_aliases
alias mr='python /your/path/to/AF-guided-MR/main.py'
source ~/.bash_aliases
```
next, install the required packages:
```
conda install -c conda-forge cctbx-base # if you haven't installed cctbx-base from the previous step
pip install nvidia-ml-py3 gemmi mdtraj polyleven pandarallel scikit-learn hdbscan colorama biopython
```
The main script requires the protein sequence and data path as input. These can be designated by the --csv_path and --mtz_path flags, respectively. The csv format follows the instructions from ColabFold in the 'id,sequence' format. Optional but highly recommended inputs include the UniProt ID (--uniprot_id) and copy numbers for the protein component you wish to search (--copy_numbers). For a full list of options and input format, run the script with the --help or -h flag.

The script, if needed, will invoke AF_cluster.py. This script clusters the MSA from ColabFold and sorts the resulting structures according to their RMSD to the top-ranked ColabFold model. This ensures that when Phaser runs on these models, it always tests with the order of high RMSD first.

The AF_cluster script here draws upon the work done by the authors of the original AF_Cluster project (https://github.com/HWaymentSteele/AF_Cluster). I have constructed a modified version to fit my specific needs, focusing on the DBSCAN clustering of MSA from ColabFold and subsequent model sorting. I deeply appreciate their fabulous work.

For example:
```
python main.py --csv_path your_protein_sequence.csv --mtz_path your_xray_data.mtz --uniprot_id P12345,P23456 --copy_numbers 2:2 --nproc 8
```
Replace your_protein_sequence.csv and your_xray_data.mtz with your actual input files, P12345 with your actual UniProt ID, and 8 with your desired number of processors.

## NOTE
This tool has been benchmarked on 372 PDB entries that are deemed hard problems for MR and has shown a 92% success rate at identifying the right solution (R-factors for AutoBuild/refine in reasonable range), or 97% success rate at finding a significant solution but require further human evaluation. It is designed to handle difficult cases where the predicted structure varies significantly from the final solution. It is not a replacement for human evaluation, however, it is a proper tool to greatly speed up the process of MR. 

The full tested entry list is at resources/tested_cases.txt. Test cases results are available upon request, because the size of the results are too large to be uploaded here. 

If you experience difficulty with the tool, or simply want to try your case but does not want to work with the full installation process, feel free to contact me at: af.guided.mr at gmail.com. I am happy to help you with your case and improve the tool.