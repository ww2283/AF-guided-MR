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
import psutil

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


def get_available_cores():
    """This function returns a recommended number of CPU cores to use for a process, e.g. Phaser."""
    total_cores = os.cpu_count()
    # Get overall CPU usage percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    used_cores = int(total_cores * (cpu_percent / 100))
    available_cores = total_cores - used_cores
    # Ensure at least 4 core is used
    recommended_cores = max(4, int(0.5 * available_cores))
    return recommended_cores

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

        if info.used <= 1000 * 1024 * 1024:
            logging.info(f"GPU {device_id} is available with {info.used / (1024**2)} MiB in use.")
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
    The input is a folder containing subfolders with refinement logs and/or autobuild logs.
    *For autobuild folders, this sub_folder_path is ./autobuild/AutoBuild_run_?_, 
    *and for refinement folders, this sub_folder_path is ./refine/refine_???.
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
    end of function definitions; start of main logic for extract_rfactors
    """
    # Check for autobuild log
    autobuild_log_path = os.path.join(sub_folder_path, 'AutoBuild_run_1_1.log')
    if os.path.exists(autobuild_log_path):
        log_refine_path = find_refinement_log_file(autobuild_log_path)
        if log_refine_path:
            r_work, r_free, extracted = extract_rfactors_from_refinement_log(log_refine_path)
        if not extracted:
            r_work, r_free, extracted = extract_rfactors_from_autobuild_log(autobuild_log_path)
    
    # Check for refinement log
    if not extracted:
        refine_folder_path = os.path.join(sub_folder_path)
        r_work, r_free, extracted = extract_rfactors_from_refine_log(refine_folder_path)
    
    return r_work, r_free    

def rfactors_from_phenix_refine(pdb_path, data_path, refine_output_root, nproc):
    # *refine_output_root is output_root + "/refine"
    # List all existing folders in the refine_output_root directory
    existing_folders = [f for f in os.listdir(refine_output_root) if os.path.isdir(os.path.join(refine_output_root, f))]

    # Filter out folders that match the pattern refine_???
    refine_folders = [f for f in existing_folders if re.match(r'refine_\d{3}', f)]

    # Extract the numeric part of the folder names and find the maximum number
    max_num = 0
    for folder in refine_folders:
        num = int(folder.split('_')[1])
        if num > max_num:
            max_num = num

    # Increment the maximum number by 1 to get the next folder number
    next_num = max_num + 1

    # Format the new folder number to be three digits
    new_folder_name = f"refine_{next_num:03d}"

    # Create the new folder
    refinement_folder = os.path.join(refine_output_root, new_folder_name)
    os.makedirs(refinement_folder, exist_ok=True)

    # check for the existence of the refinement_data.mtz file in the refine_output_root
    if os.path.exists(os.path.join(refine_output_root, "refinement_data.mtz")):
        data_path = os.path.join(refine_output_root, "refinement_data.mtz")

    # Initialize the phenix_refine_cmd with base parameters
    phenix_refine_cmd = [
        "phenix.refine",
        pdb_path,
        data_path,
        "strategy=rigid_body+individual_sites+individual_adp",
        "main.number_of_macro_cycles=8",
        f"nproc={nproc}",
        "tncs_correction=True",
        "ncs_search.enabled=True",
        "pdb_interpretation.allow_polymer_cross_special_position=True",
        "pdb_interpretation.clash_guard.nonbonded_distance_threshold=None",
        "output.write_eff_file=False",
        "output.write_def_file=False",
        "output.write_geo_file=False",
    ]

    def run_phenix_refine(cmd):
        formatted_cmd = " ".join(cmd)
        logging.info(f"Running Phenix refine with the following command into {refinement_folder}: {formatted_cmd}")
        # Verify that none of the elements in the command are None
        if any(elem is None for elem in cmd):
            raise ValueError("One or more elements in phenix_refine_cmd are None.")
        process = subprocess.Popen(cmd, cwd=refinement_folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr

    fixed_errors = set()
    max_iterations = 5

    for iteration in range(max_iterations):
        returncode, stdout, stderr = run_phenix_refine(phenix_refine_cmd)

        if returncode == 0:
            # Success
            break

        re_run = False

        # Check for errors and adjust the command accordingly
        combined_output = stdout + stderr

        if ("Multiple equally suitable arrays of observed xray data found" in combined_output) and ('multiple_arrays' not in fixed_errors):
            logging.warning("Multiple equally suitable arrays of observed X-ray data found. Extracting the first possible choice.")
            match = re.search(r'Possible choices:\s*(.*)', combined_output, re.DOTALL)
            if match:
                first_choice = match.group(1).split('\n')[0].strip()
                phenix_refine_cmd.append(f"refinement.input.xray_data.labels={first_choice}")
                fixed_errors.add('multiple_arrays')
                re_run = True
            else:
                logging.error("Could not extract array choice.")
                raise RuntimeError("Phenix refine failed due to multiple arrays issue.")

        if ("Atoms at special positions are within rigid groups" in combined_output) and ('atoms_special_positions' not in fixed_errors):
            logging.warning("Atoms at special positions are within rigid groups. Changing strategy to individual_sites+individual_adp and adding --overwrite.")
            # Modify the strategy argument
            for i, arg in enumerate(phenix_refine_cmd):
                if arg.startswith("strategy="):
                    phenix_refine_cmd[i] = "strategy=individual_sites+individual_adp"
                    break
            else:
                # Strategy argument not found, add it
                phenix_refine_cmd.append("strategy=individual_sites+individual_adp")
            if "--overwrite" not in phenix_refine_cmd:
                phenix_refine_cmd.append("--overwrite")
            fixed_errors.add('atoms_special_positions')
            re_run = True

        if (("R-free flags not compatible" in combined_output) or ("missing flag" in combined_output)) and ('r_free_flags' not in fixed_errors):
            logging.warning("R-free flags not compatible or missing flag. Generating new flags with fraction=0.05 and max_free=500.")
            phenix_refine_cmd.extend([
                "xray_data.r_free_flags.generate=True",
                "xray_data.r_free_flags.fraction=0.05",
                "xray_data.r_free_flags.max_free=500",
            ])
            if "--overwrite" not in phenix_refine_cmd:
                phenix_refine_cmd.append("--overwrite")
            fixed_errors.add('r_free_flags')
            re_run = True

        if re_run:
            continue  # Retry with the adjusted command

        # If no known errors can be fixed, raise an error
        logging.error(f"Phenix refine failed with return code {returncode}.")
        logging.error(stdout)
        logging.error(stderr)
        raise RuntimeError("Phenix refine failed.")

    else:
        # Exceeded maximum iterations
        logging.error("Exceeded maximum number of iterations.")
        raise RuntimeError("Phenix refine failed after maximum attempts.")

    logging.info(f"Phenix refine finished for {os.path.basename(pdb_path)}.")
    logging.info(f"Refinement output directory: {os.path.basename(refinement_folder)}")

    r_work, r_free = extract_rfactors(refinement_folder)
    logging.info(f"R_work: {r_work}, R_free: {r_free} for {os.path.basename(pdb_path)}")

    mtz_files = glob.glob(os.path.join(refinement_folder, "*_data.mtz"))
    if mtz_files and os.path.exists(mtz_files[0]):
        shutil.move(mtz_files[0], os.path.join(refine_output_root, "refinement_data.mtz"))

    return r_work, r_free, refinement_folder

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