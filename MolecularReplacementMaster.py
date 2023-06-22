#!/home/wei/anaconda3/envs/colabfold152/bin/python

import argparse
import csv
import os
import sys
import io
import subprocess
import time
import threading
import nvidia_smi
from iotbx import file_reader
from mmtbx.scaling.matthews import matthews_rupp
from mmtbx.scaling.twin_analyses import get_twin_laws
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Chain
from itertools import groupby
import glob
import gemmi
import shutil
import re
import os.path
import requests
from pathlib import Path
import logging
from colorama import Fore, Style, init
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelname == "SUCCESS":
            record.levelname = "\033[1;32mSUCCESS\033[0m"
        elif record.levelname == "FAIL":
            record.levelname = "\033[1;31mFAIL\033[0m"
        elif record.levelname == "ERROR":
            record.levelname = "\033[1;31mERROR\033[0m"
        elif record.levelname == "WARNING":
            record.levelname = "\033[1;33mWARNING\033[0m"
        return super().format(record)

logging.addLevelName(25, "SUCCESS")
logging.addLevelName(45, "FAIL")

log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = CustomFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler('automated_structure_solvation.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

# Define custom logging functions
def success(msg, *args, **kwargs):
    logging.log(25, msg, *args, **kwargs)

def fail(msg, *args, **kwargs):
    logging.log(45, msg, *args, **kwargs)

logging.success = success
logging.fail = fail


af_cluster_script_path = "/maintank/Wei/automation.colabfold/test45/AF_cluster.py"

def parse_args():
    parser = argparse.ArgumentParser(description="Molecular replacement and model building using protein sequence and reduced X-ray diffraction data.\n")
    parser = argparse.ArgumentParser(description="Currently support single chain protein sequence from a single protein. Stay tuned for multi chain and multimer support.\n")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file containing protein sequence.")
    parser.add_argument("--mtz_path", type=str, required=True, help="Path to the input MTZ file containing reduced X-ray diffraction data.")
    parser.add_argument("--uniprot_id", type=str, required=False, default=None, help="The Uniprot ID for your sequence, will be used for difficult cases if provided. Otherwise will be automatically determined.")
    parser.add_argument("--nproc", type=int, required=False, default=8, help="Number of processors to use (default: 8).")
    parser.add_argument("--colabfold_data_path", default=None, help="Path to the ColabFold data folder.")

    return parser.parse_args()

def read_sequence_from_csv(csv_path):
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        if header != ['id', 'sequence']:
            raise ValueError("CSV header should be 'id,sequence'")
        row = next(reader)
        structure_name, sequence = row
    return structure_name, sequence

def create_structure_directory(structure_name):
    os.makedirs(structure_name, exist_ok=True)
    return os.path.abspath(structure_name)

def get_available_gpu():
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    logging.info(f"Total GPUs: {device_count}")

    for device_id in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        used_memory_fraction = info.used / info.total
        logging.info(f"GPU {device_id}: {info.used / (1024**2)} MiB used ({used_memory_fraction * 100:.2f}% used)")

        if used_memory_fraction <= 1/15:
            logging.info(f"GPU {device_id} is available")
            return device_id

    logging.info("No available GPUs found")
    return None

def wait_for_available_gpu():
    while True:
        logging.info("Checking for available GPUs...")
        device_id = get_available_gpu()
        if device_id is not None:
            logging.info(f"Using GPU {device_id}")
            return device_id
        logging.info("Waiting for 1 minute before checking again...")
        time.sleep(60)

def run_colabfold(input_csv, output_dir, num_models=1, num_recycle=5, amber=True, use_gpu_relax=True):
    device_id = wait_for_available_gpu()

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    colabfold_command = [
        "colabfold_batch",
        "--num-models", str(num_models),
        "--num-recycle", str(num_recycle),
        input_csv,
        output_dir,
    ]
    if amber:
        colabfold_command.append("--amber")
    if use_gpu_relax:
        colabfold_command.append("--use-gpu-relax")

    subprocess.run(colabfold_command, check=True)

def has_colabfold_finished(output_dir, structure_name):
    done_file = os.path.join(output_dir, f"{structure_name}.done.txt")
    return os.path.exists(done_file)

def mark_colabfold_finished(output_dir):
    done_file = os.path.join(output_dir, f"{structure_name}.done.txt")
    with open(done_file, "w") as f:
        f.write("ColabFold finished\n")

def get_uniprot_id_from_sequence(sequence):
    # Perform a BLAST search against the SwissProt database
    result_handle = NCBIWWW.qblast("blastp", "swissprot", sequence)

    # Parse the BLAST result
    blast_records = NCBIXML.parse(result_handle)

    # Find the top hit's accession number
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            print("Alignment:", alignment)  # Add this line to inspect the alignment object
            
            uniprot_id = alignment.accession.split(".")[0]
            print("UniProt ID:", uniprot_id)  # Add this line to inspect the UniProt ID

            return uniprot_id


        
def find_pdb_file(input_dir, structure_name):
    pdb_files = glob.glob(f"{input_dir}/{structure_name}*.pdb")
    if not pdb_files:
        error_message = f"No PDB file found with pattern '{structure_name}*.pdb'"
        logging.error(error_message)
        raise FileNotFoundError(error_message)
    return pdb_files[0]

def process_pdb_file(input_pdb, b_factor_cutoff, output_pdb_path):
    # Remove lines not starting with 'ATOM' from the input PDB file    
    with open(input_pdb, 'r') as f_in, open(os.path.splitext(input_pdb)[0] + '_filtered.pdb', 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)

    intermediate_pdb = os.path.splitext(input_pdb)[0] + '_filtered.pdb'

    # Initialize a PDBParser object
    parser = PDBParser()

    cmd = f"phenix.pdbtools {intermediate_pdb} remove='element H or bfactor < {b_factor_cutoff}' output.filename={output_pdb_path}"
    subprocess.run(cmd, shell=True)
    trimmed = parser.get_structure("protein", output_pdb_path)
    # extract the residue numbers and corresponding residues into a list
    residues = [(residue.id[1], residue) for residue in trimmed.get_residues()]

    # use groupby to identify isolated pieces of residue numbers
    isolated_pieces = []
    for k, g in groupby(enumerate(residues), lambda i_x:i_x[0]-i_x[1][0]):
        group = list(map(lambda i_x:i_x[1], g))
        if len(group) >= 1:
            if len(group) <= 5:
                # delete the corresponding residues from the structure
                for residue in group:
                    chain = residue[1].get_parent()
                    chain.detach_child(residue[1].id)
            else:
                isolated_pieces.append(group)
    model = list(trimmed.get_models())[0]
    chain = model['A']

    # Set b-factors
    for residue in chain:
        for atom in residue:
            bfactor_value = 8 * np.pi**2 * 1.5 * np.exp(4 * (0.7 - atom.bfactor / 100)) / 3
            atom.bfactor = bfactor_value
    # write the modified structure to a new PDB file
    io = PDBIO()
    io.set_structure(trimmed)
    io.save(output_pdb_path)

def analyze_asu_and_solvent_content(mtz_file, sequence=None):
    mtz_object = file_reader.any_file(mtz_file).file_object
    miller_arrays = mtz_object.as_miller_arrays()

    # Assuming the first Miller array corresponds to the experimental data
    data = miller_arrays[0]
    crystal_symmetry = data.crystal_symmetry()

    n_residues = None
    n_bases = None
    sleep_time = 100
    if sequence:
        logging.info(f"sequence is : {sequence}")
        # Assuming the sequence contains only protein residues (no DNA/RNA)
        n_residues = len(sequence)
        logging.info(f"Number of residues: {n_residues}")
        sleep_time = int(n_residues / 3)
        if sleep_time < 100:
            sleep_time = 100
        logging.info(f"In case AF_cluster will be run, the time interval between each AF_cluster run will be {sleep_time} seconds")

    # Run matthews_rupp
    xtriage_analysis = matthews_rupp(
        crystal_symmetry=crystal_symmetry,
        n_residues=n_residues,
        n_bases=n_bases
    )

    # Print and return ASU and solvent content analysis
    # Redirect stdout to a buffer
    temp_stdout = io.StringIO()
    sys.stdout = temp_stdout

    # Run xtriage_analysis
    xtriage_analysis.show(out=sys.stdout)

    # Reset stdout to the original
    sys.stdout = sys.__stdout__

    # Get the output from the buffer
    xtriage_output = temp_stdout.getvalue()

    # Log the output
    logging.info(f"Xtriage analysis output:\n{xtriage_output}")

    # Extract the most probable solvent content and number of molecules
    solvent_content = xtriage_analysis.solvent_content
    num_molecules = xtriage_analysis.n_copies

    logging.info(f"Most probable solvent content: {solvent_content * 100:.2f}%")
    logging.info(f"Most probable number of molecules: {num_molecules}")

    return xtriage_analysis, solvent_content, num_molecules, sleep_time

def run_twin_analysis(mtz_file):
    mtz_object = file_reader.any_file(mtz_file).file_object
    miller_arrays = mtz_object.as_miller_arrays()

    # Assuming the first Miller array corresponds to the experimental data
    data = miller_arrays[0]

    # Get twin laws
    twin_laws = get_twin_laws(data)
    if twin_laws:
        logging.warning("Data is probably twinned, you may want to run twin refinement.")
        logging.warning(f"Twin laws: {twin_laws}")
        
        # Write the first twin law to the 'twin_law.params' file
        first_twin_law = twin_laws[0]
        with open("twin_law.params", "w") as twin_law_file:
            twin_law_file.write(f"refinement.twinning.twin_law=\"{first_twin_law}\"")
        
        return twin_laws
    else:
        logging.info("Data is not twinned")
        return None


# Add this function after the read_sequence_from_csv() function
def write_sequence_to_fasta(sequence, fasta_filename):
    with open(fasta_filename, "w") as fasta_file:
        fasta_file.write(">sequence\n")
        fasta_file.write(sequence)

# Add this function after the write_sequence_to_fasta() function
def get_space_group(mtz_path):
    mtz = gemmi.read_mtz_file(mtz_path)
    space_group = mtz.spacegroup
    return space_group.hm

def get_high_resolution(mtz_path):
    mtz = gemmi.read_mtz_file(mtz_path)
    return mtz.resolution_high()

# Add this function after the get_space_group() function
def generate_phaser_params(params_filename, hklin, seq_file, solvent_content, space_group, coordinates, num_molecules):
    with open(params_filename, "w") as f:
        f.write("phaser {\n")
        f.write("  mode = MR_AUTO\n")
        f.write(f"  hklin = {hklin}\n")
        f.write(f"  seq_file = {seq_file}\n")
        f.write(f"  composition.solvent = {solvent_content}\n")
        f.write(f"  crystal_symmetry.space_group = \"{space_group}\"\n")
        f.write("  ensemble {\n")
        f.write("    model_id = ensemble_0\n")
        f.write("    coordinates {\n")
        f.write(f"      pdb = {coordinates}\n")
        f.write("      identity = 90.0\n")
        f.write("    }\n")
        f.write("  }\n")
        f.write("  search {\n")
        f.write("    ensembles = ensemble_0\n")
        f.write(f"    copies = {num_molecules}\n")
        f.write("  }\n")
        f.write("  keywords {\n")
        f.write("    general {\n")
        f.write(f"      jobs = {args.nproc}\n")
        f.write("    }\n")
        f.write("  }\n")
        f.write("}\n")

def prepare_domain_ensembles(input_pdb, domains, domain_output_dir, sequence_length):
    # first prepare the domain isolated ensembles 
    # domains = get_prioritized_domains(uniprot_id)
    Path(domain_output_dir).mkdir(parents=True, exist_ok=True)
    # Calculate the threshold length (20% of the input sequence length)
    threshold_length = 0.2 * sequence_length
    logging.info(f"Threshold length for a domain to be used: {threshold_length}")
    sorted_domains = sorted(domains, key=lambda x: x['end'] - x['start'], reverse=True)
    # Adjust domain boundaries
    adjusted_domains = adjust_domain_boundaries(sorted_domains, sequence_length)
    logging.info(f"List of adjusted domains: {adjusted_domains}")
    
    ensemble_files = []
    
    # Extract the domains from the PDB file and save them as ensembles
    for i, domain in enumerate(adjusted_domains):
        domain_length = domain['end'] - domain['start'] + 1

        # Check if the domain length is greater than or equal to 20% of the input sequence length
        if domain_length >= threshold_length:
            logging.info(f"Domain {i+1}: {domain['start']} - {domain['end']}")
            logging.info(f"Domain {i+1} will be used for MR with length: {domain_length}")
            output_file = os.path.join(domain_output_dir, f"ensemble_{i+1}.pdb")
            ensemble_files.append(output_file)
            extract_domain_from_pdb(input_pdb, domain['start'], domain['end'], output_file)
        else:
            # If the domain length is less than 20% of the input sequence length, stop processing further domains
            logging.info(f"Domain {i+1}: {domain['start']} - {domain['end']}")
            logging.info(f"Domain length: {domain_length} less than {threshold_length}. Skipping this domain.")
            continue
    
    return ensemble_files

def get_domains_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    model = structure[0]

    domains = []
    prev_residue = None
    chain_start = None
    domain_id = 1

    for residue in model.get_residues():
        residue_id = residue.get_id()[1]
        if not re.match("^[A-Za-z]+$", residue.get_resname()):
            continue

        if prev_residue is not None and residue_id - prev_residue > 1:
            domains.append({
                "accession": f"CHAIN_BREAK_{domain_id}",
                "name": f"Domain {domain_id}",
                "start": chain_start,
                "end": prev_residue
            })
            domain_id += 1
            chain_start = residue_id

        if chain_start is None:
            chain_start = residue_id

        prev_residue = residue_id

    domains.append({
        "accession": f"CHAIN_BREAK_{domain_id}",
        "name": f"Domain {domain_id}",
        "start": chain_start,
        "end": prev_residue
    })

    return domains

def generate_alternative_phaser_params(params_filename, hklin, seq_file, solvent_content, space_group, ensemble_pdbs, num_molecules):
    with open(params_filename, "w") as f:
        f.write("phaser {\n")
        f.write("  mode = MR_AUTO\n")
        f.write(f"  hklin = {hklin}\n")
        f.write(f"  seq_file = {seq_file}\n")
        f.write(f"  composition.solvent = {solvent_content}\n")
        f.write(f"  crystal_symmetry.space_group = \"{space_group}\"\n")
        
        # Write ensemble blocks
        for ensemble_pdb in ensemble_pdbs:
            # ensemble_id = os.path.splitext(os.path.basename(ensemble_pdb))[0]
            f.write("  ensemble {\n")
            f.write(f"    model_id = {ensemble_pdb}\n")
            f.write("    coordinates {\n")
            f.write(f"      pdb = {ensemble_pdb}\n")
            f.write("      identity = 90.0\n")
            f.write("    }\n")
            f.write("  }\n")

        # Write separate search blocks for each ensemble
        for ensemble_pdb in ensemble_pdbs:
            # ensemble_id = os.path.splitext(os.path.basename(ensemble_pdb))[0]
            f.write("  search {\n")
            f.write(f"    ensembles = {ensemble_pdb}\n")
            f.write(f"    copies = {num_molecules}\n")
            f.write("  }\n")

        f.write("  keywords {\n")
        f.write("    general {\n")
        f.write(f"      jobs = {args.nproc}\n")
        f.write("    }\n")
        f.write("  }\n")
        f.write("}\n")

# Add this function after the generate_phaser_params() function
def run_phaser_molecular_replacement(params_filename):
    cmd = f"phenix.phaser {params_filename}"
    with open(os.devnull, "w") as devnull:
        subprocess.run(cmd, stdout=devnull, stderr=subprocess.STDOUT, shell=True, text=True)

def run_phaser_molecular_replacement_async(params_filename):
    phaser_cmd = ["phenix.phaser", f"{params_filename}"]
    phaser_process = subprocess.Popen(phaser_cmd)
    return phaser_process

def is_phaser_successful(phaser_output_dir):
    log_files = ["PHASER.log", os.path.join(phaser_output_dir, "PHASER.log")]

    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if "EXIT STATUS: SUCCESS" in content:
                    return True
    return False

def phaser_files_exist_in_output_dir(phaser_output_dir):
    current_dir_files = ["PHASER.log", "PHASER.1.pdb", "PHASER.1.mtz"]
    output_dir_files = [f"{phaser_output_dir}/{file}" for file in current_dir_files]
    return all(os.path.exists(file) for file in output_dir_files)

def phaser_files_exist_in_current_dir(current_dir):
    current_dir_files = ["PHASER.log", "PHASER.1.pdb", "PHASER.1.mtz"]
    return any(os.path.exists(file) for file in current_dir_files)

def handle_phaser_output(output_dir, ensemble_pdbs=None):
    success = False
    if all(os.path.exists(file) for file in ["PHASER.log", "PHASER.1.pdb", "PHASER.1.mtz"]) or all(os.path.exists(os.path.join(output_dir, file)) for file in ["PHASER.log", "PHASER.1.pdb", "PHASER.1.mtz"]):
        for file in glob.glob("PHASER.*"):
            shutil.move(file, f"{output_dir}/{file}")

        if is_phaser_successful(output_dir):
            tfz = get_final_tfz(output_dir)
            # Check if tfz is not none and greater than 8.0
            if tfz is not None and float(tfz) >= 8.0:
                success = True
                if ensemble_pdbs is None:
                    logging.info(f"Phaser molecular replacement with input model was successful. Check log file for details at {output_dir}/PHASER.log")
                else:
                    logging.success(f"Phaser molecular replacement alternative mode with ensemble models {ensemble_pdbs} was successful. Check log file for details at {output_dir}/PHASER.log")
            else:
                logging.warning(f"Phaser molecular replacement has low tfz ({tfz}) and may not be reliable. Check log file for details at {output_dir}/PHASER.log")
        else:
            logging.warning(f"Phaser molecular replacement may have failed. Check log file for details at {output_dir}/PHASER.log")
    else:
        for file in glob.glob("PHASER.*"):
            shutil.move(file, f"{output_dir}/{file}")
        logging.warning(f"Phaser molecular replacement may have failed because not all output files are present. Check log file for details at {output_dir}/PHASER.log")

    return success

def check_valid_partial_solutions(phaser_log_path):
    with open(phaser_log_path, "r") as f:
        lines = f.readlines()

    inside_solution_block = False
    valid_partial_solutions = []
    highest_tfz = 0.0

    for line in lines:
        if "Solution" in line and "written to PDB file" in line:
            inside_solution_block = True
            continue

        if inside_solution_block and line.strip() == "":
            inside_solution_block = False

        if inside_solution_block and "SOLU 6DIM ENSE" in line:
            tfz_part = line.split("#")[-1]
            if "TFZ" in tfz_part:
                tfz = float(tfz_part.split("==")[-1])
                valid_partial_solutions.append((line, tfz))
                if tfz > highest_tfz:
                    highest_tfz = tfz

    return valid_partial_solutions, highest_tfz


def get_prioritized_domains(uniprot_id):
    def is_overlapping(domain1, domain2, overlap_threshold):
        shared_start = max(domain1['start'], domain2['start'])
        shared_end = min(domain1['end'], domain2['end'])
        shared_length = max(0, shared_end - shared_start + 1)
        domain1_length = domain1['end'] - domain1['start'] + 1
        domain2_length = domain2['end'] - domain2['start'] + 1
        
        overlap_ratio1 = shared_length / domain1_length
        overlap_ratio2 = shared_length / domain2_length
        
        return overlap_ratio1 >= overlap_threshold or overlap_ratio2 >= overlap_threshold

    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot_id}?format=json"
    response = requests.get(url)

    if response.status_code != 200:
        logging.error(f"Error: Unable to fetch data for Uniprot ID {uniprot_id}")
        return []

    data = response.json()

    all_domains = []
    hs_domains = []
    other_domains = []
    max_protein_length = 0

    for result in data["results"]:
        metadata = result["metadata"]
        for protein in result["proteins"]:
            protein_length = protein["protein_length"]
            max_protein_length = max(max_protein_length, protein_length)
            for entry_protein_location in protein["entry_protein_locations"]:
                for fragment in entry_protein_location["fragments"]:
                    start = fragment["start"]
                    end = fragment["end"]
                    domain = {
                        "accession": metadata["accession"],
                        "name": metadata["name"],
                        "start": start,
                        "end": end,
                        "type": metadata["type"]
                    }
                    all_domains.append(domain)
                    if metadata["type"] == "homologous_superfamily":
                        hs_domains.append(domain)
                    else:
                        other_domains.append(domain)

    if hs_domains:
        # Calculate the coverage of homologous_superfamily domains
        coverage = sum([domain["end"] - domain["start"] + 1 for domain in hs_domains]) / max_protein_length

        if coverage >= 0.8:
            prioritized_domains = hs_domains
        else:
            prioritized_domains = hs_domains + other_domains
    else:
        prioritized_domains = other_domains

    filtered_domains = []
    for hs_domain in prioritized_domains:
        if hs_domain['type'] == 'homologous_superfamily':
            filtered_domains.append(hs_domain)
        else:
            overlapping = False
            for other_domain in prioritized_domains:
                if other_domain['type'] == 'homologous_superfamily':
                    if is_overlapping(hs_domain, other_domain, 0.8):
                        overlapping = True
                        break
            if not overlapping:
                filtered_domains.append(hs_domain)

    return filtered_domains

def extract_domain_from_pdb(pdb_file, start, end, output_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    # Create a new structure for the domain
    domain_structure = structure.copy()
    domain_model = domain_structure[0]

    # Remove all chains from the domain structure
    for chain in domain_model.get_chains():
        domain_model.detach_child(chain.id)

    # Iterate over the chains and residues, and add residues within the domain boundaries to the domain structure
    for chain in model:
        domain_chain = Chain.Chain(chain.id)
        for residue in chain:
            if start <= residue.id[1] <= end:
                domain_chain.add(residue.copy())
        if len(domain_chain) > 0:
            domain_model.add(domain_chain)

    # Save the domain structure to a new PDB file
    io = PDBIO()
    io.set_structure(domain_structure)
    io.save(output_file)

def adjust_domain_boundaries(domains, sequence_length):
    sorted_domains = sorted(domains, key=lambda x: x['start'])
    adjusted_domains = []
    for i, domain in enumerate(sorted_domains):
        start = domain['start']
        end = domain['end']
        domain_length = end - start + 1

        if i > 0:
            prev_domain = adjusted_domains[i - 1]
            prev_end = prev_domain['end']

            # Check for overlaps
            if start <= prev_end:
                current_ratio = abs((domain_length / sequence_length) - 0.2)
                prev_ratio = abs(((prev_end - prev_domain['start'] + 1) / sequence_length) - 0.2)

                if current_ratio > prev_ratio:
                    start = prev_end + 1
                else:
                    prev_domain['end'] = start - 1

        adjusted_domains.append({
            "accession": domain["accession"],
            "name": domain["name"],
            "start": start,
            "end": end
        })

    return adjusted_domains

def get_final_tfz(phaser_output_dir):
    log_files = ["PHASER.log", f"{phaser_output_dir}/PHASER.log"]

    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if "EXIT STATUS: SUCCESS" in content:
                    lines = content.split("\n")
                    last_refined_tfz_line = None
                    for line in lines:
                        if 'Refined TF/TFZ equivalent' in line:
                            last_refined_tfz_line = line
                    
                    if last_refined_tfz_line:
                        parts = last_refined_tfz_line.split("/")
                        tfz_part = parts[2].strip()
                        tfz = float(tfz_part.split()[0])  # Split by whitespace and pick the first float
                        return tfz
    return None

def run_af_cluster(msa_file, reference_pdb):
    af_cluster_script_path = "/maintank/Wei/automation.colabfold/test45/AF_cluster.py"
    result = subprocess.run([sys.executable, af_cluster_script_path, msa_file, reference_pdb], capture_output=True, text=True)

    # Print the output of the AF_cluster script
    print("AF_cluster.py stdout:")
    print(result.stdout)
    print("AF_cluster.py stderr:")
    print(result.stderr)

    return result.returncode == 0


def remove_duplicates_from_file(file_path):
    # Read the file and store the lines in a list
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Deduplicate the lines
    unique_lines = list(set(lines))

    # Write the unique lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(unique_lines)

def identify_disulfide_bonds(structure, distance_cutoff=2.1):
    cysteines = [residue for residue in structure.get_residues() if residue.get_resname() == "CYS"]
    potential_bonds = []

    for i, cys1 in enumerate(cysteines):
        for cys2 in cysteines[i+1:]:
            sg1 = cys1["SG"]
            sg2 = cys2["SG"]
            distance = sg1 - sg2

            if distance <= distance_cutoff:
                potential_bonds.append((cys1, cys2, distance))

    return potential_bonds


def disulfide_bonds_string(structure, distance_cutoff=2.2):
    potential_bonds = identify_disulfide_bonds(structure, distance_cutoff)
    
    if not potential_bonds:
        return None

    bond_string = " ".join(
        f"{cys1.get_id()[1]}:{cys2.get_id()[1]}"
        for cys1, cys2, _ in potential_bonds
    )

    if not bond_string:
        return None

    return f"-MR::disulf {bond_string}"

def remove_zero_occupancy_atoms(input_pdb_file, output_pdb_file):
    # First, save the header information
    header = []
    with open(input_pdb_file, 'r') as f:
        for line in f:
            if re.match("^(TITLE|REMARK|HELIX|SHEET|SSBOND|LINK|CRYST1|ORIGX1|ORIGX2|ORIGX3|SCALE1|SCALE2|SCALE3)\s*.*$", line):
                header.append(line)
            else:
                break

    # Parse the PDB file and remove 0 occupancy atoms
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb_file)

    for model in structure:
        for chain in model:
            for residue in chain:
                zero_occ_atoms = [atom for atom in residue.get_atoms() if atom.get_occupancy() == 0]
                for atom in zero_occ_atoms:
                    residue.detach_child(atom.get_id())

    # Write the header and the updated structure to the output file
    io = PDBIO()
    io.set_structure(structure)
    with open(output_pdb_file, 'w') as f:
        f.writelines(header)
        io.save(f)

def get_cpu_usage(pid):
    cmd = f"ps -p {pid} -o %cpu"
    output = subprocess.check_output(cmd, shell=True, text=True)
    cpu_usage = float(output.splitlines()[1].strip())
    return cpu_usage

# # the following function sets is dedicated to the mr_rosetta procedure:
def find_list_of_jobs_running_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "LIST_OF_JOBS_RUNNING":
                yield os.path.join(root, file)

def process_unhealthy_file(file_path):
    final_count_pattern = r"FINAL COUNT:\s+(\d+)"
    missing_job_pattern = r"MISSING\s*$"
    missing_job_dir = None
    # missing_job_line = None
    final_count = None

    with open(file_path, "r") as file:
        for line in file:
            if not final_count:
                match = re.search(final_count_pattern, line)
                if match:
                    final_count = int(match.group(1))
            if not missing_job_dir:
                if re.search(missing_job_pattern, line):
                    missing_job_line = line.strip().split()[1]  # Extract the path of the MISSING log file
                    missing_job_dir = os.path.dirname(missing_job_line)  # Get the directory containing the log file

    return final_count, missing_job_dir

def create_go_on_file(directory, final_count):
    go_on_file = os.path.join(directory, "GO_ON")
    if not os.path.exists(go_on_file):
        logging.warning(f"Created GO_ON file in {directory} with final count {final_count}.")
        with open(go_on_file, "w") as go_on:
            go_on.write(str(final_count))
    else:
        logging.warning(f"GO_ON file already exists in {directory}.")

def check_list_of_jobs_running_files(group_directory):
    for file_path in find_list_of_jobs_running_files(group_directory):
        final_count, missing_job_dir = process_unhealthy_file(file_path)
        if final_count and missing_job_dir:
            create_go_on_file(missing_job_dir, final_count)

def monitor_job(job_log, threshold=10, duration=1800, check_interval=600):
    info_file = job_log.replace("RUN_FILE", "INFO_FILE").rstrip(".log")
    with open(info_file, "r") as info:
        group_directory = os.path.dirname(info.readline().strip())

    info_files = [f for f in os.listdir(group_directory) if f.startswith("INFO_FILE")]

    if len(info_files) <= 1:
        logging.info(f"Skipping group_directory {group_directory} as the INFO_FILE_? count is less or equal than 1.")
        return
    if 'GROUP_OF_RESCORE_' in group_directory:
        logging.info(f"Skipping group_directory {group_directory} as it contains 'GROUP_OF_RESCORE_'.")
        return

    try:
        pids = subprocess.check_output(f"pgrep -f {info_file}", shell=True, text=True).strip().splitlines()
        pid = int(pids[0])
        time_below_threshold = 0
        go_on_created = False

        while True:
            cpu_usage = get_cpu_usage(pid)
            logging.info(f"Job {job_log} with PID {pid} has CPU usage {cpu_usage}%. Monitored for {time_below_threshold} seconds.")

            if cpu_usage < threshold:
                time_below_threshold += check_interval
            else:
                time_below_threshold = 0

            if time_below_threshold >= duration:
                if not go_on_created:
                    unhealthy_files = list(filter(lambda x: x != os.path.join(group_directory, 'LIST_OF_JOBS_RUNNING'), find_list_of_jobs_running_files(group_directory)))
                    if unhealthy_files:
                        for file_path in unhealthy_files:
                            final_count, missing_job_dir = process_unhealthy_file(file_path)
                            if final_count and missing_job_dir:
                                create_go_on_file(missing_job_dir, final_count)
                        time_below_threshold = 0
                    else:
                        final_count, missing_job_dir = process_unhealthy_file(os.path.join(group_directory, 'LIST_OF_JOBS_RUNNING'))
                        if final_count and missing_job_dir:
                            create_go_on_file(missing_job_dir, final_count)
                            go_on_created = True
                else:
                    break
            time.sleep(check_interval)

    except subprocess.CalledProcessError:
        logging.info(f"Could not find PID for job {job_log} in {group_directory}")

def monitor_mr_rosetta_logs(mr_rosetta_out, threshold=10, duration=1800, check_interval=600):
    global stop_monitoring
    
    while not os.path.exists(mr_rosetta_out):
        time.sleep(5)  # Wait for 5 seconds before checking again

    # Initialize the file position and size variables
    file_position = 0
    file_size = os.path.getsize(mr_rosetta_out)

    while True:
        # If the file has grown since the last check
        if file_size > file_position:
            with open(mr_rosetta_out, "r") as outfile:
                # Seek to the last read position
                outfile.seek(file_position)

                # Read the new content and process it
                lines = outfile.readlines()

                job_logs = []
                for line in reversed(lines):
                    if "Starting job" in line:
                        job_log = line.split()[-1]
                        job_logs.append(job_log)
                    if "Splitting work into" in line:
                        break

                logging.info(f"Found {len(job_logs)} job logs in the most recent block:")
                for job_log in job_logs:
                    logging.info(f"  - {job_log}")

                # Create and start threads to monitor each job log
                threads = []
                for job_log in job_logs:
                    t = threading.Thread(target=monitor_job, args=(job_log, threshold, duration, check_interval))
                    t.start()
                    threads.append(t)

                # Wait for all threads to finish
                for t in threads:
                    t.join()

                # Update the file position to the current position
                file_position = outfile.tell()

        # Update the file size
        file_size = os.path.getsize(mr_rosetta_out)

        # Wait for a short period before checking again
        time.sleep(119)


def run_mr_rosetta(mtz_path, pdb_path, fasta_path, nproc, space_group_str, copies, disulf_string="", twinning=None):
    with open('mr_rosetta_phenix.params', 'w') as f:
        f.write(f"refinement.pdb_interpretation.allow_polymer_cross_special_position=True")
        # f.write('\n' + 'refinement.input.xray_data.r_free_flags.disable_suitability_test=True')
        if twinning:
            with open('twin_law.params', 'r') as twin_file:
                twin_contents = twin_file.read()
                f.write('\n' + twin_contents)
    cmd = (
        f"phenix.mr_rosetta mr_rosetta.input_files.data={mtz_path} "
        f"mr_rosetta.input_files.search_models='{pdb_path}' "
        f"mr_rosetta.rescore_mr.relax=false "
        f"mr_rosetta.rosetta_rebuild.rosetta_models={nproc} "
        f"mr_rosetta.crystal_info.ncs_copies={copies} "
        f"mr_rosetta.crystal_info.space_group='{space_group_str}' "
        f"mr_rosetta.place_model.use_all_plausible_sg=False "
        f"mr_rosetta.place_model.model_already_placed=true "
        f"mr_rosetta.control.nproc={nproc} "
        f"mr_rosetta.input_files.seq_file={fasta_path} "
        f"mr_rosetta.input_files.refinement_params='mr_rosetta_phenix.params'"
    )

    # if disulf_string:
    #     cmd += f" mr_rosetta.control.rosetta_command='{disulf_string}'"

    # Run the command and redirect the output to a file
    with open("mr_rosetta.out", "w") as outfile:
        subprocess.run(cmd, shell=True, stdout=outfile)


def is_mr_rosetta_successful(mr_rosetta_out_path):
    if not os.path.exists(mr_rosetta_out_path):
        return False, None, None, None

    with open(mr_rosetta_out_path, "r") as mr_rosetta_out_file:
        contents = mr_rosetta_out_file.read()

    if "Best solution:" in contents:
        best_solution_index = contents.index("Best solution:")
        contents_after_best_solution = contents[best_solution_index:]

        model_path_line = next(line for line in contents_after_best_solution.splitlines() if "MODEL:" in line)
        map_coeffs_line = next(line for line in contents_after_best_solution.splitlines() if "MAP COEFFS" in line)

        model_path = model_path_line.split("MODEL:")[1].strip()
        map_coeffs_path = map_coeffs_line.split("MAP COEFFS")[1].strip()

        return True, model_path, map_coeffs_path, mr_rosetta_out_path
    elif "Sorry: mr_rosetta failed..." in contents:
        log_path_line = next(line for line in contents.splitlines() if "Please see log file for full error message" in line)
        log_path = log_path_line.split("Please see log file for full error message")[0].strip()

        return False, None, None, log_path

    return False, None, None, None


def save_csv_report(output_file, sequence_length, run_time, resolution, tfz_score, successful_phaser_run, successful_phaser_output_dir, mr_rosetta_success):
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['sequence_length', 'run_time', 'resolution', 'tfz_score', 'successful_phaser_run', 'successful_phaser_output_dir', 'mr_rosetta_success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'sequence_length': sequence_length,
            'run_time': run_time,
            'resolution': resolution,
            'tfz_score': tfz_score,
            'successful_phaser_run': successful_phaser_run,
            'successful_phaser_output_dir': successful_phaser_output_dir,
            'mr_rosetta_success': mr_rosetta_success
        })

def get_next_pdb_entry(file_path, used_pdbs):
    with open(file_path, "r") as file:
        lines = file.readlines()
        # skip the first line
        lines = lines[1:]
        for line in lines:
            pdb = line.split(",")[0]
            if pdb not in used_pdbs:
                return pdb
    return None


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    structure_name, sequence = read_sequence_from_csv(args.csv_path)
    output_dir = create_structure_directory(structure_name)
    # Fetch the Uniprot ID if it's not supplied
    if args.uniprot_id is None:
        logging.warning("Uniprot ID not supplied. It's recommended to supply that value whenever possible. Now trying to deduce it from the sequence...")
        args.uniprot_id = get_uniprot_id_from_sequence(sequence)
        if args.uniprot_id is None:
            logging.warning("Could not deduce Uniprot ID from the sequence. Please supply it manually.")
            exit(1)
        else:
            logging.success(f"Deduced Uniprot ID: {args.uniprot_id}")

    # Run colabfold if it has not been run before
    if not has_colabfold_finished(output_dir, structure_name):
        run_colabfold(args.csv_path, output_dir)
        mark_colabfold_finished(output_dir)
    else:
        logging.warning("Skipping colabfold since it has already been run.")
    
    tobeprocessed_pdb_paths = glob.glob(f"{output_dir}/{structure_name}_relaxed_rank_001*.pdb")
    tobeprocessed_pdb_path = tobeprocessed_pdb_paths[0] if tobeprocessed_pdb_paths else None
    os.makedirs("step_trimmed", exist_ok=True)
    processed_pdb_path = "step_trimmed/ranked_0_plddt40.pdb"

    # Process the generated PDB file
    if not os.path.exists(processed_pdb_path):
        if tobeprocessed_pdb_path is not None:
            for b_factor_cutoff in [40, 60]:
                output_pdb_path = f"step_trimmed/ranked_0_plddt{b_factor_cutoff}.pdb"
                process_pdb_file(tobeprocessed_pdb_path, b_factor_cutoff, output_pdb_path)
        else:
            logging.warning("No PDB file found for processing.")
    else:
        logging.warning("Skipping processing PDB file as it already exists.")


    # Analyze ASU and solvent content
    xtriage_analysis, solvent_content, num_molecules, sleep_time = analyze_asu_and_solvent_content(args.mtz_path, sequence)

    # Run twin analysis
    twin_results = run_twin_analysis(args.mtz_path)

    # Save the sequence as a FASTA file
    fasta_filename = f"{output_dir}/{structure_name}.fasta"
    write_sequence_to_fasta(sequence, fasta_filename)

    # Get the space group
    space_group_str = get_space_group(args.mtz_path)

    # Generate the phaser.params file
    params_filename = f"phaser.params"
    if os.path.exists(params_filename):
        os.remove(params_filename)

    generate_phaser_params(params_filename, args.mtz_path, fasta_filename, solvent_content, space_group_str, processed_pdb_path, num_molecules)

    default_phaser_output_dir = f"phaser_output"
    alternative_phaser_output_dir = f"alternative_phaser_output"
    current_dir = os.getcwd()

    # Run phaser molecular replacement with plddt40 model
    alternative_phaser_success = False
    AF_cluster_success = False
    valid_partial_solutions = False
    default_phaser_success = False
    AF_cluster_phaser_output_dir = False

    if os.path.exists(default_phaser_output_dir):
        logging.info("default phaser output directory already exists. default phaser mode won't be run.")
        logging.warning(f"if you want to run default phaser mode, please delete the directory {default_phaser_output_dir} and rerun the script.")
    else:
        os.makedirs(default_phaser_output_dir)
        logging.info(f"running phaser molecular replacement with default mode, see parameters used in {params_filename}...")
        
        run_phaser_molecular_replacement(params_filename)

    
    default_phaser_success = handle_phaser_output(default_phaser_output_dir)
    
    if default_phaser_success:
        logging.success(f"default_phaser_success: {default_phaser_success}") # type: ignore
        default_tfz_score = get_final_tfz(default_phaser_output_dir)
        logging.success(f"tfz score for default phaser run: {default_tfz_score}")
    else:
        logging.warning(f"default_phaser_success: {default_phaser_success}") # type: ignore
        default_tfz_score = None
        # Run phaser molecular replacement with alternative mode
        os.makedirs("domains", exist_ok=True)
        domain_output_dir = f"domains"
        plddt60_pdb_file = "step_trimmed/ranked_0_plddt60.pdb"
        domains = get_prioritized_domains(args.uniprot_id) or get_domains_from_pdb(plddt60_pdb_file)
        ensemble_files = []
        if domains:
            ensemble_files = prepare_domain_ensembles(processed_pdb_path, domains, domain_output_dir, len(sequence))
            logging.info(f"the following domains were found and will be used for alternative phaser molecular replacement: {ensemble_files}")
        
            alternative_params_filename = f"alternative_phaser.params"
            if os.path.exists(alternative_params_filename):
                os.remove(alternative_params_filename)

            generate_alternative_phaser_params(alternative_params_filename, args.mtz_path, fasta_filename, solvent_content, space_group_str, ensemble_files, num_molecules)

            # now run alternative phaser mr
            alternative_phaser_output_dir = f"alternative_phaser_output"
            if os.path.exists(alternative_phaser_output_dir):
                logging.info("alternative phaser may have been run. skipping this alternative run.")
                logging.warning(f"if you wish to run alternative phaser again, please delete the directory {alternative_phaser_output_dir} and try again.")                
            else:
                os.makedirs(alternative_phaser_output_dir)
                logging.info(f"running phaser molecular replacement with alternative mode, see parameters used in {alternative_params_filename}...")
                run_phaser_molecular_replacement(alternative_params_filename)
            alternative_phaser_success = handle_phaser_output(alternative_phaser_output_dir, ensemble_files)               
        else:             
            logging.info("We don't have any domain information for now. So we will try to run phaser with AF_cluster mode.")

        alternative_phaser_log_path = os.path.join(alternative_phaser_output_dir, "PHASER.log")
        
        if os.path.exists(alternative_phaser_log_path):

            if alternative_phaser_success:
                alternative_tfz_score = get_final_tfz(alternative_phaser_output_dir)
            else:
                valid_partial_solutions, alternative_partial_tfz_score = check_valid_partial_solutions(alternative_phaser_log_path)
                if valid_partial_solutions and alternative_partial_tfz_score >= 8.0:
                    logging.success(f"Valid partial solutions found with TFZ score: {alternative_partial_tfz_score}. Check {alternative_phaser_log_path} for details.")
                    valid_partial_solutions = True
                elif valid_partial_solutions and alternative_partial_tfz_score < 8.0:
                    logging.warning(f"Partial solutions found with TFZ score: {alternative_partial_tfz_score} below 8. Check {alternative_phaser_log_path} for details.")
                    valid_partial_solutions = False                
                else:
                    logging.warning("No valid partial solutions found.")
        else:
            logging.warning(f"{alternative_phaser_log_path} not found. Proceeding to AF_cluster mode.")

        if not alternative_phaser_success and not valid_partial_solutions:    
            ########################################
            ## run AF_cluster mode
            ########################################
            logging.info("Trying AF_cluster mode.")
            AF_cluster_success = False

            if os.path.exists("AF_cluster_selectives") and len(os.listdir("AF_cluster_selectives")) > 0:
                logging.warning("AF_cluster_selectives folder is not empty. Skipping AF_cluster.")
            else:
                logging.info("AF_cluster_selectives directory not found or empty. Running AF_cluster.")
                os.makedirs("AF_cluster_selectives", exist_ok=True)
                ranked_0_filtered_pdb = f"{output_dir}/ranked_0_filtered.pdb"
                
                if os.path.exists(ranked_0_filtered_pdb):
                    reference_pdb = ranked_0_filtered_pdb
                elif len(tobeprocessed_pdb_paths) > 0:
                    reference_pdb = tobeprocessed_pdb_paths[0]
                else:
                    logging.error("No reference PDB file available.")
                    raise Exception("No reference PDB file available.")
                
                logging.info(f"Calling run_af_cluster with arguments: {output_dir}/{structure_name}.a3m, {reference_pdb}")
                af_cluster_process = subprocess.Popen([sys.executable, af_cluster_script_path, f"{output_dir}/{structure_name}.a3m", reference_pdb])

                # Monitor rmsd_ranking.csv file and run phaser on new PDB entries
                used_pdbs = set()
                rmsd_ranking_file = "AF_cluster_selectives/rmsd_ranking.csv"
                af_cluster_finish_file = "AF_cluster_selectives/ALL_AFCLUSTER_DONE"
                predictions_folder = "predictions"
                AF_cluster_phaser_output_dir = "AF_cluster_phaser_output"
                AF_cluster_treated_pdb_dir = "AF_cluster_plddt_treated_pdbs"

                while not AF_cluster_success:
                    if os.path.exists(rmsd_ranking_file):
                        next_pdb = get_next_pdb_entry(rmsd_ranking_file, used_pdbs)
                        if next_pdb is None and os.path.exists(af_cluster_finish_file):
                            logging.warning("All PDB entries from rmsd_ranking.csv have been tested with no success.")
                            break
                        while next_pdb is None and not os.path.exists(af_cluster_finish_file):
                            time.sleep(30)
                            next_pdb = get_next_pdb_entry(rmsd_ranking_file, used_pdbs)
                        if next_pdb:
                            used_pdbs.add(next_pdb)
                            tobeprocessed_pdb_path = os.path.join("AF_cluster_selectives", next_pdb)
                            print(f"The selected tobeprocessed_pdb_path: {tobeprocessed_pdb_path}")
                            cluster_number = re.search("cluster_(\d+)_", tobeprocessed_pdb_path).group(1)
                            processed_pdb_path = f"{AF_cluster_treated_pdb_dir}/cluster_{cluster_number}_plddt40.pdb"
                            os.makedirs(AF_cluster_treated_pdb_dir, exist_ok=True)
                            process_pdb_file(tobeprocessed_pdb_path, 40, processed_pdb_path)

                            # Generate params file
                            params_filename = f"phaser_params_cluster_{cluster_number}.txt"
                            generate_phaser_params(params_filename, args.mtz_path, fasta_filename, solvent_content, space_group_str, processed_pdb_path, num_molecules)
                            os.makedirs(AF_cluster_phaser_output_dir, exist_ok=True)
                            # Run phaser
                            logging.info(f"Running phaser molecular replacement with AF_cluster model {processed_pdb_path}...")
                            phaser_process = run_phaser_molecular_replacement_async(params_filename)
                            phaser_log_path = f"PHASER.log"
                            while phaser_process.poll() is None:
                                if os.path.exists(phaser_log_path):
                                    log_size_mb = os.path.getsize(phaser_log_path) / (1024 * 1024)
                                    if log_size_mb > 5:
                                        logging.warning("Terminating phaser run due to an abnormally long time.")
                                        phaser_process.terminate()
                                        break
                                time.sleep(20)  # Adjust the sleep time as needed

                            AF_cluster_mr_success = handle_phaser_output(AF_cluster_phaser_output_dir)
                            if AF_cluster_mr_success:
                                AF_cluster_tfz_score = get_final_tfz(AF_cluster_phaser_output_dir)
                                logging.success(f"Phaser molecular replacement with AF_cluster model {processed_pdb_path} was successful with tfz: {AF_cluster_tfz_score}.")
                                logging.info("Thus the AF_cluster will be stopped.")
                                halt_file_path = os.path.join(predictions_folder, 'HALT')

                                with open(halt_file_path, 'w') as halt_file:
                                    halt_file.write('')

                                af_cluster_process.terminate()
                                AF_cluster_success = True
                                break
                            else:
                                logging.warning("This AF_cluster phaser run was not successful.")
                                AF_cluster_tfz_score = None
                            # logging.info(f"The PDBs that have been used: {used_pdbs}")

                    if os.path.exists(f"{predictions_folder}/HALT"):
                        logging.warning("Halting AF_cluster.")
                        af_cluster_process.terminate()
                        break

                    time.sleep(sleep_time)  # Adjust the sleep time as needed

    if default_phaser_success:
        successful_phaser_output_dir = default_phaser_output_dir
        best_tfz_score = default_tfz_score
    elif alternative_phaser_success:
        successful_phaser_output_dir = alternative_phaser_output_dir
        best_tfz_score = alternative_tfz_score
    elif valid_partial_solutions:
        successful_phaser_output_dir = alternative_phaser_output_dir
        best_tfz_score = alternative_partial_tfz_score
    elif AF_cluster_success:
        successful_phaser_output_dir = AF_cluster_phaser_output_dir
        best_tfz_score = AF_cluster_tfz_score
    else:
        logging.fail("All phaser attempts failed. Program failed to find a successful MR result. However, you may find some useful information by examinine each of the phaser output directories.")
        raise Exception("Program failed to find a successful MR result. However, you may find some useful information by examinine each of the phaser output directories.")

    mr_solution = os.path.join(successful_phaser_output_dir, "PHASER.1.pdb")


    # Now you can use the mr_solution file for the subsequent MR-Rosetta jobs

    is_successful = "N/A"
    parser_ssbond = PDBParser()
    mr_rosetta_structure = parser_ssbond.get_structure("structure_id", mr_solution)
    logging.info(f"the mr_solution is {mr_solution}, which will be used to check for the presence of ssbonds")
    disulf_string = disulfide_bonds_string(mr_rosetta_structure)
    if disulf_string:
        logging.warning(f"ssbonds are present and is: {disulf_string}. The mr_rosetta will be run without this value. However, if you see problem because the disulfide bonds were not kept, you rerun mr_rosetta with this disulfide bonds definition.")
    else:
        logging.info("ssbonds are not present in the structure.")

    if mr_solution is not None:
        high_resolution = get_high_resolution(args.mtz_path)
        if high_resolution <= 2.5:
            if not os.path.exists("mr_rosetta.out"):
                logging.info(f"Data resolution is {high_resolution}. Running MR-Rosetta with {mr_solution}.")
                
                # Remove 0 occupancy atoms
                new_mr_solution = mr_solution.replace(".pdb", "_no_zero_occupancy.pdb")
                remove_zero_occupancy_atoms(mr_solution, new_mr_solution)
                mr_solution = new_mr_solution            

                # Start the monitor_mr_rosetta function in a separate thread
                monitor_thread = threading.Thread(target=monitor_mr_rosetta_logs, args=("mr_rosetta.out",))
                monitor_thread.daemon = True  # Set the thread as a daemon thread
                monitor_thread.start()

                run_mr_rosetta(args.mtz_path, mr_solution, fasta_filename, args.nproc, space_group_str, num_molecules, twinning=twin_results)
                mr_rosetta_out_path = "mr_rosetta.out"
                is_successful, model_path, map_coeffs_path, log_path = is_mr_rosetta_successful(mr_rosetta_out_path)

                if is_successful:
                    logging.success(f"MR-Rosetta was successful. Model: {model_path}, Map Coeffs: {map_coeffs_path}.")
                    logging.info(f"You can check the log file at {log_path} for details.")
                else:
                    if log_path:
                        logging.warning(f"MR-Rosetta failed. Check the log file at {log_path} for details.")
                    else:
                        logging.fail("MR-Rosetta failed.")                    
            else:
                logging.warning("MR-Rosetta has already been run. Skipping.")
        else:
            logging.warning("Data resolution is worse than 2.5 Angstroms. Skipping MR-Rosetta.")

    logging.info("Auto processing is done. You can check the log file for details at automated_structure_solvation.log.")

    run_time = time.time() - start_time

    global stop_monitoring
    stop_monitoring = True

    phaser_run_names = ['default', 'alternative', 'valid_partial', 'AF_cluster']

    successful_phaser_run = phaser_run_names[
        [
            default_phaser_success,
            alternative_phaser_success,
            valid_partial_solutions,
            AF_cluster_success
        ].index(True)
    ]

    successful_phaser_output_dir = [
        default_phaser_output_dir,
        alternative_phaser_output_dir,
        alternative_phaser_output_dir,
        AF_cluster_phaser_output_dir
    ][
        [
            default_phaser_success,
            alternative_phaser_success,
            valid_partial_solutions,
            AF_cluster_success
        ].index(True)
    ]
    tfz_score = best_tfz_score
    resolution = get_high_resolution(args.mtz_path)
    mr_rosetta_success = is_successful if mr_solution and high_resolution <= 2.5 else None
    sequence_length = len(read_sequence_from_csv(args.csv_path)[1])
    csv_report_filename = f"report.csv"
    save_csv_report(csv_report_filename, sequence_length, run_time, resolution, tfz_score, successful_phaser_run, successful_phaser_output_dir, mr_rosetta_success)