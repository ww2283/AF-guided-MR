

import os
import argparse
import numpy as np
import pandas as pd
import mdtraj as md
from sklearn.cluster import DBSCAN
from glob import glob
from Bio import SeqIO
import subprocess
import shutil
import time
import nvidia_smi
import logging
from Bio.PDB import PDBParser, Select
from Bio.PDB.Superimposer import Superimposer
import pycuda.driver as cuda

logger = logging.getLogger(__name__)

# Include functions from ClusterMSA.py, utils.py, and CalculateModelFeatures.py

# Functions from ClusterMSA.py
def cluster_msa(msa_file):
    # Load MSA file
    IDs, seqs = load_fasta(msa_file)

    # Remove lowercase letters in alignment
    seqs = remove_lowercase_letters_in_alignment(seqs)

    # Calculate maximum length of sequences in MSA
    max_len = max([len(seq) for seq in seqs])

    # Encode sequences
    encoded_seqs = encode_seqs(seqs, max_len=max_len)

    # Scan for optimal epsilon
    min_samples = 3
    optimal_eps = None
    max_n_clusters = -1
    eps_candidates = np.arange(3, 20.5, 0.5)
    n_clusters = []

    for eps in eps_candidates:
        testset = encoded_seqs[np.random.choice(encoded_seqs.shape[0], int(0.25 * encoded_seqs.shape[0]), replace=False), :]
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        temp_cluster_labels = clusterer.fit_predict(testset)
        n_clust = len(set(temp_cluster_labels))
        if eps > 10 and n_clust == 1:
            break
        n_clusters.append(n_clust)

    optimal_eps = eps_candidates[np.argmax(n_clusters)]
    print("Optimal epsilon: %.2f" % optimal_eps)
    # Cluster sequences using the optimal epsilon

    clusterer = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(encoded_seqs)

    return cluster_labels

def remove_lowercase_letters_in_alignment(seqs):
    cleaned_seqs = []
    for seq in seqs:
        cleaned_seq = ''.join([char for char in seq if not char.islower()])
        cleaned_seqs.append(cleaned_seq)
    return cleaned_seqs


# Functions from utils.py
def lprint(string, f):
    print(string)
    f.write(string + "\n")


def load_fasta(fil):
    seqs, IDs = [], []
    with open(fil) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = "".join([x for x in record.seq])
            IDs.append(record.id)
            seqs.append(seq)
    return IDs, seqs

def write_fasta(IDs, seqs, outfile):
    with open(outfile, "w") as f:
        for ID, seq in zip(IDs, seqs):
            f.write(f">{ID}\n{seq}\n")

def write_clusters(cluster_labels, IDs, seqs, output_folder):
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1:
            continue

        cluster_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == label]
        cluster_IDs = [IDs[i] for i in cluster_indices]
        cluster_seqs = [seqs[i] for i in cluster_indices]

        # Ensure the first sequence of the input MSA is included in the output files
        if IDs[0] not in cluster_IDs:
            cluster_IDs.insert(0, IDs[0])
            cluster_seqs.insert(0, seqs[0])

        output_path = os.path.join(output_folder, f"cluster_{label:03d}.a3m")
        write_fasta(cluster_IDs, cluster_seqs, outfile=output_path)


def encode_seqs(seqs, max_len=None, alphabet=None):
    if alphabet is None:
        alphabet = "ACDEFGHIKLMNPQRSTVY-"

    if max_len is None:
        max_len = max([len(seq) for seq in seqs])
    
    arr = np.zeros([len(seqs), max_len, len(alphabet)])
    for j, seq in enumerate(seqs):
        for i, char in enumerate(seq):
            for k, res in enumerate(alphabet):
                if char == res:
                    arr[j, i, k] += 1
    return arr.reshape([len(seqs), max_len * len(alphabet)])



def consensusVoting(seqs):
    consensus = ""
    residues = "ACDEFGHIKLMNPQRSTVWY-"
    n_chars = len(seqs[0])
    for i in range(n_chars):
        baseArray = [x[i] for x in seqs]
        baseCount = np.array([baseArray.count(a) for a in list(residues)])
        vote = np.argmax(baseCount)
        consensus += residues[vote]

    return consensus

def get_nvml_gpu_info():
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    gpu_info_list = []

    for i in range(device_count):
        try:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            uuid = nvidia_smi.nvmlDeviceGetUUID(handle).decode('utf-8')
            pci_info = nvidia_smi.nvmlDeviceGetPciInfo(handle)
            bus_id = pci_info.busId.decode('utf-8')  # e.g., '0000:21:00.0'
            name = nvidia_smi.nvmlDeviceGetName(handle).decode('utf-8')
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem_info.used // (1024 * 1024)  # Convert to MiB
            mem_total = mem_info.total // (1024 * 1024)  # Convert to MiB

            gpu_info_list.append({
                'nvml_index': i,
                'uuid': uuid,
                'bus_id': bus_id,
                'name': name,
                'mem_used': mem_used,
                'mem_total': mem_total,
            })
        except nvidia_smi.NVMLError as err:
            logger.warning(f"Failed to get information for GPU {i}: {err}")
            continue
    nvidia_smi.nvmlShutdown()
    return gpu_info_list

def get_pycuda_gpu_info():
    cuda.init()
    device_count = cuda.Device.count()
    gpu_info_list = []

    for i in range(device_count):
        dev = cuda.Device(i)
        name = dev.name()
        compute_capability = dev.compute_capability()  # Tuple (major, minor)
        pci_bus_id = dev.pci_bus_id()  # e.g., '0000:21:00.0'

        gpu_info_list.append({
            'pycuda_index': i,
            'name': name,
            'compute_capability': compute_capability,
            'bus_id': pci_bus_id,
        })
    return gpu_info_list

def get_gpu_info():
    nvml_info = get_nvml_gpu_info()
    pycuda_info = get_pycuda_gpu_info()

    # Create a mapping from bus_id to compute_capability
    busid_to_compute_capability = {
        gpu['bus_id']: gpu['compute_capability'] for gpu in pycuda_info
    }

    # Merge the compute capability into the NVML info
    for gpu in nvml_info:
        bus_id = gpu['bus_id']
        compute_capability = busid_to_compute_capability.get(bus_id, (0, 0))
        gpu['compute_capability'] = compute_capability[0] + compute_capability[1] / 10.0

    return nvml_info

def select_gpu(gpu_info_list, max_used_mem=1000):
    """Selects the available GPU with the highest compute capability and memory usage below the max_used_mem threshold."""
    # Filter GPUs based on memory usage
    available_gpus = [gpu for gpu in gpu_info_list if gpu['mem_used'] <= max_used_mem]

    if not available_gpus:
        return None

    # Sort by compute capability (descending) and memory usage (ascending)
    available_gpus.sort(key=lambda gpu: (-gpu['compute_capability'], gpu['mem_used']))

    return available_gpus[0]

def set_cuda_visible_devices(gpu_uuid):
    """Sets CUDA_VISIBLE_DEVICES to the GPU's UUID."""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid

def get_available_gpu():
    gpu_info_list = get_gpu_info()
    if not gpu_info_list:
        logger.error("No GPU information retrieved.")
        return None

    # # Log the GPU information
    # for gpu in gpu_info_list:
    #     logger.info(
    #         f"GPU {gpu['nvml_index']} ({gpu['name']}, UUID: {gpu['uuid']}, Bus ID: {gpu['bus_id']}) - "
    #         f"Compute Capability: {gpu['compute_capability']}, Memory Used: {gpu['mem_used']} MiB"
    #     )

    available_gpu = select_gpu(gpu_info_list)
    if available_gpu:
        # logger.info(
        #     f"Selected GPU {available_gpu['nvml_index']} ({available_gpu['name']}, UUID: {available_gpu['uuid']}) "
        #     f"with Compute Capability: {available_gpu['compute_capability']}"
        # )
        return available_gpu
    else:
        logger.info("No available GPUs found based on memory usage and compute capability.")
        return None

def wait_for_available_gpu():
    while True:
        available_gpu = get_available_gpu()
        if available_gpu:
            gpu_uuid = available_gpu['uuid']
            set_cuda_visible_devices(gpu_uuid)
            # logger.info(
            #     f"Using GPU UUID {gpu_uuid} ({available_gpu['name']}, Compute Capability: {available_gpu['compute_capability']})"
            # )
            return available_gpu
        logger.info("No available GPU found. Waiting for 1 minute before retrying...")
        time.sleep(60)

def run_colabfold(input_dir, output_dir, reference_pdb, num_recycle=5, num_models=1, min_mean_pLDDT=60):


    ref_parser = PDBParser()
    ref_obj = ref_parser.get_structure("reference", reference_pdb)
    
    a3m_files = glob(os.path.join(input_dir, "*.a3m"))
    a3m_files.sort()  # Sort the input files by name

    af_cluster_selectives_dir = os.path.join(output_dir, "AF_cluster_selectives")
    os.makedirs(af_cluster_selectives_dir, exist_ok=True)

    predictions_dir = os.path.join(output_dir, "predictions")

    for a3m_file in a3m_files:
        # Check for the presence of the HALT file in the predictions folder
        halt_file_path = os.path.join(predictions_dir, "HALT")
        if os.path.exists(halt_file_path):
            logger.warning("HALT file detected. Stopping ColabFold.")
            break

        a3m_basename = os.path.basename(a3m_file)
        cluster_name = os.path.splitext(a3m_basename)[0]
        output_subdir = os.path.join(predictions_dir, cluster_name)

        command = [
            "colabfold_batch",
            "--num-recycle", str(num_recycle),
            "--num-models", str(num_models),
            a3m_file,
            output_subdir,
        ]

        logger.info("Checking for available GPUs...")
        wait_for_available_gpu()

        # No need to set CUDA_VISIBLE_DEVICES here; it's already set in the environment
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())

        if result.returncode != 0:
            logger.error(f"Error running ColabFold for {a3m_file}: {result.stderr}")
        else:
            logger.info(f"ColabFold completed successfully for {a3m_file}.")
            pdb_file = glob(os.path.join(output_subdir, "*.pdb"))[0]
            if min_mean_pLDDT == 60:
                process_predicted_structure(pdb_file, ref_obj, af_cluster_selectives_dir, min_mean_pLDDT=60)
            elif min_mean_pLDDT == 40:
                process_predicted_structure(pdb_file, ref_obj, af_cluster_selectives_dir, min_mean_pLDDT=40)



# Functions from CalculateModelFeatures.py
def read_b_factor(pdb_file):
    vals = []

    with open(pdb_file, "r") as f:
        for lin in f.readlines()[1:-3]:
            if lin[12:16].strip() == "CA":
                try:
                    b_factor = float(lin[60:66].strip())
                    vals.append(b_factor)
                except ValueError as e:
                    logger.error(f"Error converting to float: {lin[60:66].strip()} in line: {lin.strip()}")
                    raise e
    return vals


def calc_rmsd(pdb_path, ref_obj, pLDDT_vector):
    parser = PDBParser()
    pdb_obj = parser.get_structure("predicted", pdb_path)

    class CASelect(Select):
        def accept_atom(self, atom):
            return atom.get_name() == "CA"

    ref_atoms = [atom for atom in ref_obj.get_atoms() if CASelect().accept_atom(atom)]
    pred_atoms = [atom for atom in pdb_obj.get_atoms() if CASelect().accept_atom(atom)]

    # Filter out CA atoms with pLDDT values greater than 65
    ref_atoms_filtered = [atom for atom, plddt in zip(ref_atoms, pLDDT_vector) if plddt > 65]
    pred_atoms_filtered = [atom for atom, plddt in zip(pred_atoms, pLDDT_vector) if plddt > 65]

    # Find common atoms
    common_atoms = set(atom.get_parent().get_id() for atom in ref_atoms_filtered) & set(atom.get_parent().get_id() for atom in pred_atoms_filtered)

    ref_atoms_common = [atom for atom in ref_atoms_filtered if atom.get_parent().get_id() in common_atoms]
    pred_atoms_common = [atom for atom in pred_atoms_filtered if atom.get_parent().get_id() in common_atoms]

    # print(f"Number of reference atoms: {len(ref_atoms_common)}, indices: {[a.get_parent().get_id() for a in ref_atoms_common]}")
    # print(f"Number of predicted atoms: {len(pred_atoms_common)}, indices: {[a.get_parent().get_id() for a in pred_atoms_common]}")

    if not ref_atoms_common or not pred_atoms_common:
        return float("inf")

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms_common, pred_atoms_common)

    return superimposer.rms


def calculate_mean_pLDDT(row):
    pLDDT_vector = row["pLDDT_vector"]
    assert isinstance(pLDDT_vector, (list, np.ndarray)), f"Unexpected type: {type(pLDDT_vector)}"
    mean_pLDDT = np.mean(pLDDT_vector)
    return mean_pLDDT

# Function to copy good predicted structures and update CSV file
def process_predicted_structure(pdb_file, ref_obj, output_dir, min_mean_pLDDT=60):
    pLDDT_vector = read_b_factor(pdb_file)
    mean_pLDDT = np.mean(pLDDT_vector)

    if mean_pLDDT >= min_mean_pLDDT:
        cluster_name = os.path.dirname(pdb_file).split(os.path.sep)[-1]        
        dest_path = os.path.join(output_dir, f"{cluster_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb")
        pdb_entry= f"{cluster_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
        shutil.copy(pdb_file, dest_path)
        pae_json_path = os.path.join(os.path.dirname(pdb_file), f"{cluster_name}_predicted_aligned_error_v1.json")
        shutil.copy(pae_json_path, output_dir)
        
        rmsd_ref = calc_rmsd(dest_path, ref_obj, pLDDT_vector)

        csv_file = os.path.join(output_dir, "rmsd_ranking.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["pdb", "mean_pLDDT", "rmsd_ref"])

        new_row = pd.DataFrame([{"pdb": pdb_entry, "mean_pLDDT": mean_pLDDT, "rmsd_ref": rmsd_ref}])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values(by="rmsd_ref", ascending=False)
        df.to_csv(csv_file, index=False)

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workflow for clustering protein sequences, running AlphaFold2, and calculating structural features.")
    parser.add_argument("msa_file", help="Path to the input MSA file.")
    parser.add_argument("reference_pdb", help="Path to the reference PDB file.")
    parser.add_argument("--root_dir", default=".", help="Path to the root directory.")
    parser.add_argument("--output_dir", default="results", help="Path to the output directory.")
    parser.add_argument("low_cutoff", default=60, type=int, help="Low cutoff for pLDDT values.")
    args = parser.parse_args()


    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler('AF_cluster.log'), logging.StreamHandler()]
    )

    logger = logging.getLogger(__name__)

    # Load MSA file
    IDs, seqs = load_fasta(args.msa_file)
    
    # Cluster the MSA
    cluster_labels = cluster_msa(args.msa_file)
    output_dir = os.path.join(args.root_dir, args.output_dir)
    # Create the output directories
    os.makedirs(output_dir, exist_ok=True)
    inference_dir = os.makedirs(os.path.join(args.root_dir, "predictions"), exist_ok=True)
    # save clustered sequences in separate FASTA files
    write_clusters(cluster_labels, IDs, seqs, output_dir)

    saving_dir = args.root_dir

    # Run ColabFold on the clustered MSA files
    if args.low_cutoff == 40:
        run_colabfold(output_dir, saving_dir , args.reference_pdb, min_mean_pLDDT=40)
    else:
        run_colabfold(output_dir, saving_dir , args.reference_pdb)
    # Create ALL_AFCLUSTER_DONE file inside AF_cluster_selectives directory
    with open(f"{saving_dir}/AF_cluster_selectives/ALL_AFCLUSTER_DONE", "w") as f:
        f.write("All AF clustering is done!")
    logger.info("Results saved in the AF_cluster_selectives folder. All AF clustering is done!")