# AF-guided MR
A Python tool that automates molecular replacement using protein sequences and x-ray diffraction data, designed especially to handle difficult cases.

This tool uses protein sequences and reduced x-ray diffraction data to automate the process of molecular replacement. It leverages the power of ColabFold for initial structure prediction and then refines the structure based on predefined modes. For high-resolution cases better than 3.5 angstroms, it uses AutoBuild to enhance and build the model. For low-resolution cases worse than 3.5 angstroms, it uses phenix.refine to run default refinement cycles for a brief and quick assessment of the molecular replacement correctness. It is specifically designed to handle difficult cases where the predicted structure varies significantly from the final solution.

## Getting Started

The tool consists of a series of Python scripts:
1. `main.py` (the main script)
2. `AF_cluster.py` (used for handling special cases)

### Prerequisites

You'll need to have the following installed:
- conda
- Python 3.7 (preferably within a Conda environment)
- PHENIX (Python-based Hierarchical ENvironment for Integrated Xtallography)
- Rosetta software suite (optional)

### Installation (for pre Ada Lovelace cards)
#### first install conda
You can follow the [official Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for detailed instructions.
After installing Conda, create a new Conda environment with Python 3.7:
```
conda create -n automatemr python=3.7
```
you can choose other names for your environment.
Activate your newly created Conda environment:
```
conda activate automatemr
```
#### install ColabFold
Although local-colabfold should work as well, I would recommend the default ColabFold, as this python function was tested with the default version.
```
pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
pip install "jax[cuda]>=0.3.8,<0.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0
conda install -c conda-forge openmm=7.5.1 pdbfixer
```
The version I tested was giving stable results so if there's difficulty getting the colabfold to work on your machine, you can consider replace the default colabofld install with:
```
pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold@5b3fc193e880cd9599f91cd16fcb1fe69f7759f2"

```
### Installation (for Ada Lovelace cards) update 09-16-2023
I recently encountered problem with above installation method after I upgrade my cards from RTX 5000 16GB to RTX 6000 Ada. After some troubleshooting I traced down the errors are due to the upgraded needs for Ada Lovelace cards. So you will probably face similar issues if you use RTX 4080 or 4090 cards.

First you need to upgrade CUDA to 11.8 or later so that your Ada Lovelace cards can use CUDA. Then you will need to upgrade python from 3.7 to at least 3.9 and jax with the right CUDA support.

There are two possible ways to solve this issue after upgrading CUDA. You can either upgrade python with `conda install --upgrade python==3.9`, which I didn't have the luck of success. You may get away if you use mamba solver, but I haven't installed mamba so I'm not sure.

The other way is to use a new conda env. so after you upgraded your CUDA to 11.8 or later (I use 11.8):
```
conda create -n automatemr python=3.9
pip install "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0
conda install -c conda-forge openmm=7.7.0 pdbfixer
pip install --upgrade dm-haiku
```
If you encounter problem when running colabfold because of pdbfixer, like `ModuleNotFoundError: No module named 'simtk.openmm.app.internal'`, then you may need to do the following:
use a text editor, like nano or vim, open your pdbfixer.py, e.g.
`nano ~/anaconda3/envs/automatemr/lib/python3.9/site-packages/pdbfixer/pdbfixer.py`
then replace every instance of `simtk.openmm` with just `openmm`.

## Usage

The main script, MolecularReplacementMaster.py, requires the protein sequence and data path as input. These can be designated by the --csv_path and --mtz_path flags, respectively. The csv format follows the instructions from ColabFold in the 'id,sequence' format. Optional inputs include the UniProt ID (--uniprot_id) and the number of processors to use (--nproc).

The script, if needed, will invoke AF_cluster.py. This script clusters the MSA from ColabFold and sorts the resulting structures according to their RMSD to the top-ranked ColabFold model. This ensures that when Phaser runs on these models, it always tests with the order of high RMSD first.

The AF_cluster script here draws upon the work done by the authors of the original AF_Cluster project (https://github.com/HWaymentSteele/AF_Cluster). I have constructed a modified version to fit my specific needs, focusing on the DBSCAN clustering of MSA from ColabFold and subsequent model sorting. I deeply appreciate their contribution to the scientific community, which has enabled us to further advance in the field of automated molecular replacement.

For example:
```
python MolecularReplacementMaster.py --csv_path your_protein_sequence.csv --mtz_path your_xray_data.mtz --uniprot_id P12345 --nproc 8
```
Replace your_protein_sequence.csv and your_xray_data.mtz with your actual input files, P12345 with your actual UniProt ID, and 8 with your desired number of processors.

## NOTE
some works to be done:
- Google Colab integration
