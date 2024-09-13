# utilities.py
import logging
import os
import csv
import shutil
import subprocess
import time
import numpy as np
import glob
import nvidia_smi

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

def setup_custom_logger(dir):
    logging.addLevelName(25, "SUCCESS")
    logging.addLevelName(45, "FAIL")

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = CustomFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(dir, 'automated_structure_solvation.log'))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler]
    )

    def success(msg, *args, **kwargs):
        logging.log(25, msg, *args, **kwargs)

    def fail(msg, *args, **kwargs):
        logging.log(45, msg, *args, **kwargs)

    logging.success = success
    logging.fail = fail

def remove_readonly(func, path, excinfo):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)
    
def create_clean_log_copy():
    """
    remove the custom formatting from the automated_structure_solvation.log file and create a cleaned copy
    """
    def remove_ansi_codes(text):
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)
    
    original_log = 'automated_structure_solvation.log'
    cleaned_log = 'cleaned_automated_structure_solvation.log'

    with open(original_log, 'r') as infile, open(cleaned_log, 'w') as outfile:
        for line in infile:
            clean_line = remove_ansi_codes(line)
            outfile.write(clean_line)

def save_csv_report(output_file, num_sequences, sequence_length, run_time, resolution, tfz_score, successful_phaser_run, successful_phaser_output_dir, reference_model_map_cc, phaser_model_map_cc, r_work, r_free, ):
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['num_sequences', 'sequence_length', 'run_time', 'resolution', 'tfz_score', 'successful_phaser_run', 'successful_phaser_output_dir', 'reference_model_map_cc', 'phaser_model_map_cc', 'r_work', 'r_free']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'num_sequences': num_sequences,
            'sequence_length': sequence_length,
            'run_time': f'{run_time:.2f}',
            'resolution': f'{resolution:.2f}',
            'tfz_score': tfz_score,
            'successful_phaser_run': successful_phaser_run,
            'successful_phaser_output_dir': successful_phaser_output_dir,
            'reference_model_map_cc': reference_model_map_cc,
            'phaser_model_map_cc': phaser_model_map_cc,
            'r_work': r_work,
            'r_free': r_free
        })

def get_cpu_usage(pid):
    cmd = f"ps -p {pid} -o %cpu"
    output = subprocess.check_output(cmd, shell=True, text=True)
    cpu_usage = float(output.splitlines()[1].strip())
    return cpu_usage

def get_available_gpu():
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    logging.info(f"Total GPUs: {device_count}")

    for device_id in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        used_memory_fraction = info.used / info.total
        logging.info(f"GPU {device_id}: {info.used / (1024**2)} MiB used ({used_memory_fraction * 100:.2f}% used)")

        if used_memory_fraction <= 1/20:
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

def create_structure_directory(structure_name):
    os.makedirs(structure_name, exist_ok=True)
    return os.path.abspath(structure_name)

def calculate_mean_plddt(pdb_file):
    vals = []

    try:
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM") and line[13:16].strip() == "CA":  # Check if line is for a C-alpha atom
                    try:
                        b_factor = float(line[60:66].strip())  # Extract B-factor (pLDDT score)
                        vals.append(b_factor)
                    except ValueError:
                        # Handle the case where conversion to float fails
                        continue

        if not vals:
            raise ValueError("No pLDDT scores found in the file.")

        mean_pLDDT = np.mean(vals)
        return mean_pLDDT
    except IOError:
        # Handle file reading errors
        print(f"Error: Unable to read the file {pdb_file}")
        return None
    except Exception as e:
        # Handle other unforeseen errors
        print(f"Error: {str(e)}")
        return None
    

import re

def extract_rfactors(sub_folder_path):

    """
    This extract_rfactors is used to extract R-factors from refinement logs and autobuild logs.
    The input is a parent folder containing subfolders with refinement logs and autobuild logs.
    """
    def find_refinement_log_file(autobuild_log_path):
        try:
            with open(autobuild_log_path, 'r') as file:
                lines = file.read()
                log_refine_path = re.search(r"log_refine: (.+\.log_refine)", lines).group(1)
                return log_refine_path
        except Exception as e:
            print(f"Error finding refinement log file in {autobuild_log_path}: {e}")
            return None

    def extract_rfactors_from_refinement_log(log_file_path):
        try:
            with open(log_file_path, 'r') as file:
                lines = file.readlines()
                last_line = lines[-1]
                r_work, r_free = re.findall(r"R\(work\) = ([\d.]+), R\(free\) = ([\d.]+)", last_line)[0]
                return float(r_work), float(r_free), True
        except Exception as e:
            print(f"Error extracting R-factors from {log_file_path}: {e}")
            return 0.00, 0.00, False

    def extract_rfactors_from_autobuild_log(autobuild_log_path):
        try:
            with open(autobuild_log_path, 'r') as file:
                lines = file.readlines()
                for line in reversed(lines):
                    if "New values of R/Rfree:" in line:
                        r_work, r_free = re.findall(r"R/Rfree:\s*([\d.]+)/\s*([\d.]+)", line)[0]
                        return float(r_work), float(r_free), True
        except Exception as e:
            print(f"Error extracting R-factors from {autobuild_log_path}: {e}")
        return 0.00, 0.00, False

    def extract_rfactors_from_refine_log(refine_folder_path): # folder from phenix.refine
        try:
            refine_log_files = glob.glob(os.path.join(refine_folder_path, '*_refine_001.log'))
            if refine_log_files:
                with open(refine_log_files[0], 'r') as file:
                    lines = file.readlines()
                    for line in reversed(lines):
                        if "Final R-work =" in line:
                            r_work, r_free = re.findall(r"Final R-work = ([\d.]+), R-free = ([\d.]+)", line)[0]
                            return float(r_work), float(r_free), True
        except Exception as e:
            print(f"Error extracting R-factors from refine log: {e}")
        return 0.00, 0.00, False

    r_work, r_free, extracted = 0.00, 0.00, False  # Default values
    """
    end of function definitions; start of main code
    """
    # Check for autobuild log
    autobuild_log_path = os.path.join(sub_folder_path, 'autobuild', 'AutoBuild_run_1_', 'AutoBuild_run_1_1.log')
    if os.path.exists(autobuild_log_path):
        log_refine_path = find_refinement_log_file(autobuild_log_path)
        if log_refine_path:
            r_work, r_free, extracted = extract_rfactors_from_refinement_log(log_refine_path)
        if not extracted:
            r_work, r_free, extracted = extract_rfactors_from_autobuild_log(autobuild_log_path)
    
    # Check for refinement log
    if not extracted:
        refine_folder_path = os.path.join(sub_folder_path, 'refine')
        r_work, r_free, extracted = extract_rfactors_from_refine_log(refine_folder_path)
    
    return r_work, r_free    

def get_autobuild_results_paths(autobuild_working_path):
    overall_best_pdb = os.path.join(autobuild_working_path, "overall_best.pdb")
    overall_best_refine_map_coeffs = os.path.join(autobuild_working_path, "overall_best_refine_map_coeffs.mtz")
    if not os.path.exists(overall_best_pdb):
        overall_best_pdb = None
    if not os.path.exists(overall_best_refine_map_coeffs):
        overall_best_refine_map_coeffs = None
    return overall_best_pdb, overall_best_refine_map_coeffs

def get_refined_pdb_and_map(refinement_folder_path):
    pdb_file = None
    map_file = None

    # Search for refined pdb file
    pdb_files = glob.glob(os.path.join(refinement_folder_path, '*_refine_001.pdb'))
    if pdb_files:
        pdb_file = pdb_files[0]

    # Search for refined map file
    map_files = glob.glob(os.path.join(refinement_folder_path, '*_refine_001.mtz'))
    if map_files:
        map_file = map_files[0]

    return pdb_file, map_file

def calculate_map_model_correlation(pdb_file, data_file, map_file, solvent_content, output_dir, reference_pdb=None, reference_map=None):
    """
    Calculate the map-model correlation using Phenix, including NCS finding, NCS averaging,
    density modification, and final correlation calculation.

    Parameters:
    pdb_file (str): Path to the PDB file, phaser pdb or autobuild/refinement pdb.
    map_file (str): Path to the map file (MTZ format).
    solvent_content (float): Solvent content percentage (as a decimal, e.g., 0.5 for 50%).
    output_dir (str): Directory where temporary files and results will be stored.

    Returns:
    float: The calculated map-model correlation value.
    """
    correlation_value = None

    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Delete the contents of the output directory
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        # Run phenix.find_ncs
        find_ncs_command = f"phenix.find_ncs {map_file} directories.temp_dir={output_dir}/temp_dir directories.output_dir={output_dir}"
        # logging.info(f"Correlation calculation command step 1: {find_ncs_command}") # development purpose logging
        subprocess.run(find_ncs_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Run phenix.ncs_average if NCS file exists
        ncs_spec = os.path.join(output_dir, "find_ncs.ncs_spec")
        if os.path.exists(ncs_spec):
            ncs_average_command = f"phenix.ncs_average {map_file} ncs_in={ncs_spec} directories.temp_dir={output_dir}/temp_dir directories.output_dir={output_dir}"
            # logging.info(f"Correlation calculation command step 2: {ncs_average_command}") # development purpose logging
            subprocess.run(ncs_average_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            map_file = os.path.join(output_dir, "ncs_average.mtz")

        # Run phenix.density_modification
        denmod_command = f"phenix.density_modification {data_file} map_coeffs_file={map_file} pdb_file={pdb_file} solvent_content={solvent_content} clean_up=true output_files.output_mtz={output_dir}/denmod.mtz directories.temp_dir={output_dir}/temp_dir"
        if os.path.exists(ncs_spec):
            denmod_command += f" ncs_file={ncs_spec}"
        # logging.info(f"Correlation calculation command step 3: {denmod_command}") # development purpose logging
        subprocess.run(denmod_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Update the map file to the density-modified one
        map_file = os.path.join(output_dir, "denmod.mtz")

        # Run phenix.get_cc_mtz_pdb to calculate correlation
        try:
            if reference_pdb is None and reference_map is None: # this is for phaser pdb or autobuild/refinement pdb
                get_cc_command = f"phenix.get_cc_mtz_pdb {map_file} {pdb_file} output_dir={output_dir}"
                # logging.info(f"Correlation calculation command step 4: {get_cc_command}") # development purpose logging
                subprocess.run(get_cc_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif reference_pdb is not None: # this is for regular cc calculation using reference pdb
                get_cc_command = f"phenix.get_cc_mtz_pdb {map_file} {reference_pdb} output_dir={output_dir}"
                # logging.info(f"Correlation calculation command step 4: {get_cc_command}") # development purpose logging
                subprocess.run(get_cc_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            # Log the exception if needed
            logging.error(f"An error occurred during map-model correlation calculation: {e}")        

        # Parse the cc.log file to extract the correlation value
        cc_log_path = os.path.join(output_dir, "cc.log")
        if os.path.exists(cc_log_path):
            with open(cc_log_path, "r") as cc_log_file:
                for line in cc_log_file:
                    if line.startswith("  Overall map correlation:"):
                        correlation_value = float(line.split()[-1])
                        break

        if (reference_pdb is None and reference_map is not None) or not os.path.exists(cc_log_path):
            """this is for cc calculation using reference map, in case the reference pdb is not available,
            or for unknown reason the cc.log file is not generated"""
            get_cc_command = f"phenix.get_cc_mtz_mtz mtz_1={map_file} mtz_2={reference_map} output_dir={output_dir}"
            subprocess.run(get_cc_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cc_log_path = os.path.join(output_dir, "offset.log")
            if os.path.exists(cc_log_path):
                with open(cc_log_path, "r") as cc_log_file:
                    for line in cc_log_file:
                        if line.startswith("Final CC of maps:"):
                            correlation_value = float(line.split()[-1])
                            break
    except Exception as e:
        # Log the exception if needed
        logging.error(f"An error occurred during map-model/map correlation calculation: {e}")

    return correlation_value