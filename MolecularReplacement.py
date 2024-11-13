import subprocess
import os
import re
import glob
import shutil
import csv
import itertools
import logging
import time
from iotbx import file_reader
from mmtbx.scaling.matthews import matthews_rupp, p_solc_calc
from mmtbx.scaling.twin_analyses import get_twin_laws
from collections import defaultdict
import io
import sys
import PDBManager

from utilities import get_available_cores

class MolecularReplacement:
    def __init__(self, pdb_manager, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.pdb_manager = PDBManager.PDBManager()
        self.terminate_flag = False  # Add class-level termination flag

    def terminate_current_run(self):
        """Safely terminate current phaser run"""
        self.terminate_flag = True

    # Add this function after the generate_phaser_params() function    
    def run_phaser_molecular_replacement_async(self, params_filename, working_dir, ignore_timeout=False):
        try:
            # Read nproc value from params file
            with open(params_filename, 'r') as f:
                params_content = f.read()
                # Look for jobs = X in the params file
                match = re.search(r'jobs\s*=\s*(\d+)', params_content)
                if match:
                    current_nproc = int(match.group(1))
                else:
                    current_nproc = 4  # fallback to default
            
            # Calculate adjusted timeout
            base_timeout = 1800  # 30 minutes for 4 cores
            base_cores = 4
            # Formula: new_timeout = base_timeout * (base_cores/current_nproc)
            # Add minimum threshold of 300 seconds (5 minutes)
            timeout_seconds = max(300, int(base_timeout * (base_cores/current_nproc)))
              
            phaser_cmd = ["phenix.phaser", f"{params_filename}"]
            phaser_process = subprocess.Popen(phaser_cmd, cwd=working_dir)
            
            phaser_log_path = os.path.join(working_dir, "PHASER.log")
            start_time = time.time()

            found_good_tfz = False  # Flag to check if TFZ condition is met
            
            while phaser_process.poll() is None:
                # Check termination flag first
                if self.terminate_flag:
                    self.logger.warning("Terminating phaser run due to external request")
                    phaser_process.terminate()
                    self.terminate_flag = False  # Reset flag
                    break

                if os.path.exists(phaser_log_path):
                    # Always check log size
                    log_size_mb = os.path.getsize(phaser_log_path) / (1024 * 1024)
                    if log_size_mb > 4:
                        self.logger.warning("Terminating phaser run due to log file size exceeding 4MB.")
                        phaser_process.terminate()
                        break

                    if not found_good_tfz:
                        # Read the last part of the log file to check for the TFZ condition
                        try:
                            with open(phaser_log_path, 'rb') as log_file:
                                log_file.seek(0, os.SEEK_END)
                                file_size = log_file.tell()
                                buffer_size = 1024 * 10  # Read last 10KB
                                if file_size > buffer_size:
                                    log_file.seek(-buffer_size, os.SEEK_END)
                                else:
                                    log_file.seek(0)
                                bytes_content = log_file.read()
                                # Decode the binary content to text
                                text_content = bytes_content.decode('utf-8', errors='ignore')
                                lines = text_content.splitlines()
                        except Exception as e:
                            self.logger.error(f"Error reading PHASER.log: {e}")
                            lines = []
                        # Check for the desired pattern in the log lines
                        for line in reversed(lines):
                            if 'SOLU 6DIM ENSE' in line:
                                match = re.search(r'#TFZ==(\d+\.\d+)', line)
                                if match:
                                    tfz_value = float(match.group(1))
                                    if tfz_value >= 8.0:
                                        found_good_tfz = True
                                        self.logger.info(f"Found TFZ value {tfz_value} >= 8.0. Allowing process to continue.")
                                        break
                        if not found_good_tfz:
                            # Check if elapsed time exceeds limit, only if ignore_timeout is False
                            if not ignore_timeout:
                                elapsed_time = time.time() - start_time
                                if elapsed_time > timeout_seconds:
                                    self.logger.warning(f"Terminating phaser run due to exceeding the {timeout_seconds} seconds time limit.")
                                    phaser_process.terminate()
                                    break
                        else:
                            # TFZ condition met, continue without checking time limit
                            pass
                    else:
                        # TFZ condition already met, continue but still check log size
                        pass  # Log size already checked at the beginning
                else:
                    # Log file does not exist yet; wait and retry
                    pass
                time.sleep(20)  # Adjust the sleep time as needed
            
            return phaser_process
        except Exception as e:
            self.logger.error(f"Error in run_phaser_molecular_replacement_async: {e}")
            return None
    
    def is_phaser_successful(self, phaser_output_dir):
        try:
            log_files = ["PHASER.log", os.path.join(phaser_output_dir, "PHASER.log")]

            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        content = f.read()
                        if "EXIT STATUS: SUCCESS" in content:
                            return True
            return False
        except Exception as e:
            self.logger.error(f"Error in is_phaser_successful: {e}")
            return False
        
    def handle_phaser_output(self, output_dir, ensemble_pdbs=None):
        try:
            success = False
            if all(os.path.exists(file) for file in ["PHASER.log", "PHASER.1.pdb", "PHASER.1.mtz"]) or all(os.path.exists(os.path.join(output_dir, file)) for file in ["PHASER.log", "PHASER.1.pdb", "PHASER.1.mtz"]):
                for file in glob.glob("PHASER.*"):
                    shutil.move(file, f"{output_dir}/{file}")
                for file in glob.glob("alternative_phaser.params"):
                    shutil.copy(file, f"{output_dir}/{file}")

                if self.is_phaser_successful(output_dir):
                    logging.info(f"Phaser molecular replacement was successful. Check log file for details at {output_dir}/PHASER.log")
                    tfz, llg = self.get_final_tfz(output_dir)
                    logging.info(f"Final TFZ score: {tfz}, LLG score: {llg}")

                    content = open(f"{output_dir}/PHASER.log", "r").read()
                    llg_threshold = 40
                    # logging.info(f"LLG threshold: {llg_threshold:.1f}")
                    # Check if tfz is not none and greater than 8.0
                    if tfz is not None and float(tfz) >= 8.0 and float(llg) >= llg_threshold:
                        success = True
                        if ensemble_pdbs is None:
                            logging.info(f"Phaser molecular replacement with input model was successful. Check log file for details at {output_dir}/PHASER.log")
                        else:
                            logging.success(f"Phaser molecular replacement alternative mode with ensemble models {ensemble_pdbs} was successful. Check log file for details at {output_dir}/PHASER.log")
                    elif tfz is not None and float (tfz) >= 8.0 and float(llg) < llg_threshold and float(llg) > 0.0:
                        # move the files into a save folder
                        save_dir = os.path.join(output_dir, "save")
                        os.makedirs(save_dir, exist_ok=True)
                        for file in itertools.chain(glob.glob(f"{output_dir}/PHASER.*"), glob.glob(f"{output_dir}/*.params")):
                            name_base = os.path.basename(file)
                            shutil.copy(file, f"{save_dir}/{name_base}")
                        logging.warning(f"Phaser molecular replacement has low llg ({llg}) but with acceptable tfz {tfz} and may not be reliable. Check log file for details at {save_dir}/PHASER.log")
                    elif (tfz is not None and float(tfz) >= 8.0 and "** SINGLE solution" in content and int(llg) == 0) or (tfz is not None and float(tfz) >= 8.0 and (int(llg) == 0 or float(llg) == 0.0)):
                        # move the files into a save folder
                        save_dir = os.path.join(output_dir, "save")
                        os.makedirs(save_dir, exist_ok=True)
                        for file in glob.glob(f"{output_dir}/PHASER.*"):
                            name_base = os.path.basename(file)
                            shutil.copy(file, f"{save_dir}/{name_base}")
                        logging.success(f"Phaser molecular replacement may have TNCS present but with acceptable tfz {tfz}. Check log file for details at {save_dir}/PHASER.log")
                        success = True
                    elif tfz is not None and 6.0 <= float(tfz) < 8.0 and "** SINGLE solution" in content:
                        save_dir = os.path.join(output_dir, "save")
                        os.makedirs(save_dir, exist_ok=True)
                        for file in glob.glob(f"{output_dir}/PHASER.*"):
                            name_base = os.path.basename(file)
                            shutil.copy(file, f"{save_dir}/{name_base}")
                        logging.warning(f"Phaser molecular replacement has moderate tfz ({tfz}) with a single solution. Check log file for details at {save_dir}/PHASER.log.\nTesting a refinement with the current solution.")
                        success = True

                    else:
                        logging.warning(f"Phaser molecular replacement has low tfz ({tfz}) and may not be reliable. Check log file for details at {output_dir}/PHASER.log")
                else:
                    logging.warning(f"Phaser molecular replacement may have failed. Check log file for details at {output_dir}/PHASER.log")
            else:
                for file in glob.glob("PHASER.*"):
                    shutil.move(file, f"{output_dir}/{file}")
                for file in glob.glob("alternative_phaser.params"):
                    shutil.copy(file, f"{output_dir}/{file}")
                logging.warning(f"Phaser molecular replacement may have failed because not all output files are present. Check log file for details at {output_dir}/PHASER.log")

            return success
        except Exception as e:
            self.logger.error(f"Error in handle_phaser_output: {e}")
            return False
        
    def get_final_tfz(self, phaser_output_dir):
        try:
            log_files = ["PHASER.log", f"{phaser_output_dir}/PHASER.log"]
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                    tolerance = 0.001

                    highest_tfz = 0.0
                    highest_llg = 0.0
                    llg_tfz_pairs = []
                    in_solu_set = False
                    tncs_present = False
                    for line in lines:
                        if "Solution" in line and "written to PDB file" in line:
                            for sol_line in lines[lines.index(line):]:
                                if sol_line.strip() == "":
                                    break  # End of solution block
                                if "SOLU SET" in sol_line:
                                    in_solu_set = True
                                    if "+TNCS" in sol_line:
                                        tncs_present = True
                                elif "SOLU SPAC" in sol_line:
                                    in_solu_set = False

                                if in_solu_set:

                                    if tncs_present and "TFZ" in sol_line:
                                        tfz_values = [float(part.split('=')[-1]) for part in sol_line.split() if 'TFZ=' in part]
                                        highest_tfz = max(highest_tfz, max(tfz_values, default=0.0))

                                    if "LLG" in sol_line and "TFZ" in sol_line:
                                        llg_tfz_pairs_found = re.findall(r'LLG=(\d+) TFZ==(\d+\.\d+)', sol_line)
                                        if llg_tfz_pairs_found:
                                            llg_tfz_pairs.extend(llg_tfz_pairs_found)
                                        # logging.info(f"llg_tfz_pairs: {llg_tfz_pairs}")

                                if "SOLU 6DIM ENSE" in sol_line and not tncs_present:
                                    tfz_part = sol_line.split("#")[-1]
                                    if "TFZ" in tfz_part:
                                        tfz = float(tfz_part.split("==")[-1])
                                        if tfz > highest_tfz:
                                            highest_tfz = tfz
                                            hightest_tfz_part = tfz_part
                                            highest_llg_tfz_pair = None
                                            for llg, tfz_scores in llg_tfz_pairs:
                                                if abs(float(tfz_scores) - highest_tfz) < tolerance:
                                                    highest_llg_tfz_pair = (llg, tfz_scores)
                                            if highest_llg_tfz_pair:
                                                highest_llg = highest_llg_tfz_pair[0]
                                                # logging.info(f"highest_llg_tfz_pair: {highest_llg_tfz_pair}")
                                            else:
                                                highest_llg = 0.0

                    if highest_tfz > 0.0 or highest_llg > 0.0:
                        return highest_tfz, highest_llg

            return None, None
        except Exception as e:
            self.logger.error(f"Error in get_final_tfz: {e}")
            return None, None    

    """
    the following functions are for multi-copy MR
    """
    
    def multi_analyze_asu_and_solvent_content(self, mtz_file, csv_file, sequence_ids, specified_copy_numbers=None):
        """
        analyze_asu_and_solvent_content() function for multi-copy MR
        """
        mw_residue = 110
        def get_mean_matthews_coeff(space_group_number):
            # Space groups by crystal system
            # Reference: International Tables for Crystallography (2016). Volume A: Space-group symmetry
            space_groups = {
                # Triclinic system (1-2)
                **dict.fromkeys(range(1, 3), 2.40), 
                # Monoclinic system (3-15)
                **dict.fromkeys(range(3, 16), 2.43), 
                # Orthorhombic system (16-74)
                **dict.fromkeys(range(16, 75), 2.51), 
                # Tetragonal system (75-142)
                **dict.fromkeys(range(75, 143), 3.04), 
                # Trigonal system (143-167)
                **dict.fromkeys(range(143, 168), 2.94), 
                # Hexagonal system (168-194)
                **dict.fromkeys(range(168, 195), 3.08), 
                # Cubic system (195-230)
                **dict.fromkeys(range(195, 231), 3.44)
            }
            return space_groups.get(space_group_number, 2.69)  # Default to general mean if not found


        def read_protein_sequences(csv_file):
            proteins = {}
            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    proteins[row['id']] = len(row['sequence'])
                    print(f"Protein {row['id']} has {len(row['sequence'])} residues.")
            return proteins

        def estimate_amino_acid_range(mtz_file):
            mtz_object = file_reader.any_file(mtz_file).file_object
            crystal_symmetry = mtz_object.as_miller_arrays()[0].crystal_symmetry()
            
            # Assuming a typical range of Vm from 2.0 to 3.5 Å³/Da for initial estimates
            min_vm, max_vm = 1.8, 3.5
            mw_residue = 110  # Average molecular weight of an amino acid residue in Daltons
            
            asu_volume = crystal_symmetry.unit_cell().volume() / crystal_symmetry.space_group().order_z()
            max_residues = asu_volume / (min_vm * mw_residue)
            min_residues = asu_volume / (max_vm * mw_residue)

            return int(min_residues), int(max_residues), asu_volume, crystal_symmetry

        def find_combinations(proteins, mtz_file, min_residues, max_residues):        
            def attempt_find_combinations(copy_ranges):
                combinations = []
                closest_combination = None
                smallest_diff = float('inf')

                for copy_nums in itertools.product(*[range(*copy_ranges[protein]) for protein in proteins]):
                    sorted_combo = sorted([(protein, copy_nums[i], proteins[protein]) for i, protein in enumerate(proteins)], key=lambda x: x[2], reverse=True)
                    total_residues = sum([chain[1] * chain[2] for chain in sorted_combo])
                        
                    if min_residues <= total_residues <= max_residues:
                        total_mw = total_residues * mw_residue
                        matthews_rupp_result = matthews_rupp(
                            crystal_symmetry=crystal_symmetry,
                            n_residues=total_residues
                        )
                        solvent_content = matthews_rupp_result.solvent_content
                        prob = p_solc_calc(solvent_content)
                        
                        matthews_coeff = cell_volume / (total_mw * z)
                            
                        combo_str = ", ".join([f"{chain[0]} x {chain[1]}" for chain in sorted_combo])
                        combinations.append((combo_str, prob, solvent_content, matthews_coeff))

                        diff_with_mean = abs(matthews_coeff - mean_matthews_coeff)
                        if diff_with_mean < smallest_diff:
                            smallest_diff = diff_with_mean
                            closest_combination = (combo_str, prob, solvent_content, matthews_coeff)

                return combinations, closest_combination

            mtz_object = file_reader.any_file(mtz_file).file_object
            crystal_symmetry = mtz_object.as_miller_arrays()[0].crystal_symmetry()
            cell_volume = crystal_symmetry.unit_cell().volume()
            z = crystal_symmetry.space_group().order_z()
            asu_volume = cell_volume / z
            mw_residue = 110  # Average molecular weight of an amino acid residue in Daltons

            space_group_number = crystal_symmetry.space_group_info().type().number()
            mean_matthews_coeff = get_mean_matthews_coeff(space_group_number)
            
            # all other normal cases: First attempt with standard copy number range
            copy_ranges = {protein: (1, 5) if length > 50 else (1, 3) for protein, length in proteins.items()}
            combinations, closest_combination = attempt_find_combinations(copy_ranges)

            # If no combinations found, try again with expanded copy number range
            if not combinations:
                expanded_copy_ranges = {protein: (1, 10) if length > 50 else (1, 5) for protein, length in proteins.items()}
                combinations, closest_combination = attempt_find_combinations(expanded_copy_ranges)

            # Handle the case when protein length is less than min_residues
            if not combinations and all(length < min_residues for length in proteins.values()):
                logging.warning("Warning: Protein length less than estimated minimum. Checking for 1 copy, potential other components like nucleic acids may be present.")
                for protein, length in proteins.items():
                    if length < min_residues:
                        matthews_rupp_result = matthews_rupp(
                            crystal_symmetry=crystal_symmetry,
                            n_residues=length
                        )
                        
                        solvent_content = matthews_rupp_result.solvent_content
                        prob = p_solc_calc(solvent_content)
                        matthews_coeff = asu_volume / length * mw_residue
                        combination_str = f"{protein} x 1"
                        combination_info = (combination_str, prob, solvent_content, matthews_coeff)
                        combinations.append(combination_info)
                
                closest_combination = min(combinations, key=lambda x: abs(x[3] - mean_matthews_coeff)) if combinations else None
                
            return mean_matthews_coeff, combinations, closest_combination
 
        
        proteins = read_protein_sequences(csv_file)
        min_residues, max_residues, asu_volume, crystal_symmetry = estimate_amino_acid_range(mtz_file)
        mean_matthews_coeff, combinations, closest_combination = find_combinations(proteins, mtz_file, min_residues, max_residues)

        if specified_copy_numbers:
            # Ensure specified copy numbers match the length of sequence IDs
            if len(sequence_ids) != len(specified_copy_numbers):
                raise ValueError("Length of specified copy numbers does not match the number of sequences.")

            # Calculate Matthews coefficient and solvent content directly from specified copy numbers
            total_residues = sum(proteins[sequence_id] * specified_copy_numbers[sequence_id] for sequence_id in sequence_ids)
            # Calculate Matthews coefficient and solvent content directly from specified copy numbers
            # total_residues = sum(proteins[protein_id] * copy_num for protein_id, copy_num in specified_copy_numbers.items())
            total_mw = total_residues * mw_residue
            asu_volume = crystal_symmetry.unit_cell().volume() / crystal_symmetry.space_group().order_z()

            z = crystal_symmetry.space_group().order_z()

            matthews_coeff = asu_volume / total_mw
            # Calculate solvent content
            solvent_content = 1 - (1.23/matthews_coeff)


            # Create a combination string from specified copy numbers
            copy_numbers_all_combinations = {}
            combo_str = ", ".join([f"{protein_id} x {copy_num}" for protein_id, copy_num in specified_copy_numbers.items()])
            specified_combination = (combo_str, None, solvent_content, matthews_coeff)
            copy_numbers_all_combinations[combo_str] = (None, solvent_content, matthews_coeff, specified_copy_numbers)

            return specified_combination, copy_numbers_all_combinations, mean_matthews_coeff          
        elif combinations:
            # Store the copy numbers for each combination
            copy_numbers_all_combinations = {}
            for combo in combinations:
                combo_str, prob, solvent_content, matthews_coeff = combo
                copy_numbers = {}
                for protein_copy_combo in combo_str.split(", "):
                    protein, copy_num = protein_copy_combo.split(" x ")
                    copy_numbers[protein] = int(copy_num)
                copy_numbers_all_combinations[combo_str] = (prob, solvent_content, matthews_coeff, copy_numbers)

            print(f"Most favorable combination: {closest_combination[0]}")
            print(f"Probability: {closest_combination[1]:.2%}, Solvent Content: {closest_combination[2]:.2f}, Matthews Coefficient: {closest_combination[3]:.2f}")
            print(f"All combinations: {copy_numbers_all_combinations}")
            # Return the most favorable combination, all combinations, and mean Matthews coefficient
            return closest_combination, copy_numbers_all_combinations, mean_matthews_coeff
        else:
            print("No suitable combination found.")
            return None, None, None
        
    def generate_phaser_params_multimer(self, params_filename, hklin, solvent_content, space_group, processed_models, copy_numbers, phaser_info, nproc):
        try:
            adjusted_nproc = get_available_cores()
            # Use adjusted_nproc if valid, otherwise fallback to passed nproc
            final_nproc = adjusted_nproc if adjusted_nproc > 0 else nproc
        except Exception as e:
            print(f"Warning: Could not get available cores: {e}. Using provided nproc value.")
            final_nproc = nproc

        with open(params_filename, "w") as f:
            f.write("phaser {\n")
            f.write("  mode = MR_AUTO\n")
            f.write(f"  hklin = {hklin}\n")
            f.write(f"  composition.solvent = {solvent_content}\n")
            f.write(f"  crystal_symmetry.space_group = \"{space_group}\"\n")
            # Update phaser_info with ensemble details
            phaser_info['ensembles'] = {}
            for idx, (protein_id, model_paths) in enumerate(processed_models.items()):
                num_copies = copy_numbers.get(protein_id, 1)
                if not isinstance(model_paths, list):
                    model_paths = [model_paths]  # Ensure model_paths is a list

                for model_idx, model_path in enumerate(model_paths):
                    if model_path is None:
                        continue
                    ensemble_id = f"{protein_id}_ensemble_{model_idx}"
                    phaser_info['ensembles'][ensemble_id] = {
                        'protein_id': protein_id,
                        'model_path': model_path,
                        'copies': num_copies
                    }

                    f.write("  ensemble {\n")
                    f.write(f"    model_id = {ensemble_id}\n")
                    f.write("    coordinates {\n")
                    f.write(f"      pdb = {model_path}\n")
                    f.write("      identity = 90.0\n")
                    f.write("    }\n")
                    f.write("  }\n")

                    f.write("  search {\n")
                    f.write(f"    ensembles = {ensemble_id}\n")
                    f.write(f"    copies = {num_copies}\n")
                    f.write("  }\n")

                    model_idx += 1

            f.write("  keywords {\n")
            f.write("    general {\n")
            f.write(f"      jobs = {final_nproc}\n")
            f.write("    }\n")
            f.write("  }\n")
            f.write("}\n")

    def deduce_missing_copies(self, phaser_log_path, phaser_info, all_combinations, mean_matthews_coeff, top_switch=True):
        found_copies = defaultdict(int)
        ensemble_keeps = self.pdb_manager.parse_phaser_log(phaser_log_path)
        protein_ensemble_keeps = defaultdict(lambda: defaultdict(int))

        # Regular expression to extract protein_id, ensemble_number, and optional copy_number
        # Examples:
        # 'protein01_ensemble_0' -> protein_id='protein01', ensemble_number='0', copy_number=None
        # 'protein01_ensemble_0[1]' -> protein_id='protein01', ensemble_number='0', copy_number='1'
        ensemble_pattern = re.compile(r"^(.*?)_ensemble_(\d+)(?:\[(\d+)\])?$")

        # Aggregate keeps by protein_id and ensemble_number (domain)
        for ensemble_id, keep in ensemble_keeps:
            if ensemble_id and keep:
                match = ensemble_pattern.match(ensemble_id)
                if match:
                    protein_id = match.group(1)
                    ensemble_number = match.group(2)
                    copy_number = match.group(3)

                    # If copy_number is specified, use it; otherwise, treat each occurrence as a separate copy
                    if copy_number:
                        # Using the copy_number as a key to ensure each copy is counted once
                        domain_id = f"{ensemble_number}[{copy_number}]"
                        protein_ensemble_keeps[protein_id][ensemble_number] += 1
                    else:
                        # Treat each occurrence as a separate copy of the domain
                        protein_ensemble_keeps[protein_id][ensemble_number] += 1
                else:
                    # Handle unexpected ensemble_id formats if necessary
                    print(f"Warning: Unrecognized ensemble_id format '{ensemble_id}'")
        
        print(f"Protein ensemble keeps: {protein_ensemble_keeps}")

        # Calculate found_copies as the minimum count across all domains for each protein
        for protein_id, ensemble_counts in protein_ensemble_keeps.items():
            # If a protein has multiple domains, take the minimum count across domains
            if len(ensemble_counts) > 1:
                found_copies[protein_id] = min(ensemble_counts.values())
            else:
                # If only one domain, take its count
                found_copies[protein_id] = next(iter(ensemble_counts.values()))
        
        print(f"Actual found copies: {found_copies}")

        # Filter combinations to find those that match the found copies
        filtered_combinations = []
        for combo_str, (prob, solvent_content, matthews_coeff, copy_numbers) in all_combinations.items():
            if all(found_copies[protein_id] <= copies for protein_id, copies in copy_numbers.items()):
                filtered_combinations.append((combo_str, copy_numbers, prob, solvent_content, matthews_coeff, abs(matthews_coeff - mean_matthews_coeff)))

        # Sort combinations by their proximity to the mean Matthews Coefficient
        filtered_combinations.sort(key=lambda x: x[5])
        # print(f"All combinations are {all_combinations},\na total of {len(all_combinations)} all combinations")
        # print(f"Filtered combinations by deduce_missing_copies: {filtered_combinations}, \na total of {len(filtered_combinations)} filtered combinations")
        if len(filtered_combinations) > 1 and top_switch:
            return filtered_combinations[1][1], filtered_combinations[1][3], found_copies # Return the copy numbers of the second closest combination
        elif len(filtered_combinations) > 1 and not top_switch:
            return filtered_combinations[0][1], filtered_combinations[0][3], found_copies # Return the copy numbers of the closest combination
        else:
            return None, None, found_copies  # Return found_copies even if no suitable second combination

    def generate_phaser_params_for_second_run(self, params_filename, hklin, updated_solvent_content, space_group, partial_pdb_path, processed_models, missing_copies, phaser_info, nproc):
        with open(params_filename, "w") as f:
            f.write("phaser {\n")
            f.write("  mode = MR_AUTO\n")
            f.write(f"  hklin = {hklin}\n")
            f.write(f"  composition.solvent = {updated_solvent_content}\n")
            f.write(f"  crystal_symmetry.space_group = \"{space_group}\"\n")

            # Ensemble for the partial solution
            f.write("  ensemble {\n")
            f.write("    model_id = ensemble_0\n")
            f.write("    solution_at_origin = True\n")
            f.write("    coordinates {\n")
            f.write(f"      pdb = {partial_pdb_path}\n")
            f.write("      identity = 90.0\n")
            f.write("    }\n")
            f.write("  }\n")
            # Ensembles and search parameters for the missing copies
            for protein_id, model_paths in processed_models.items():
                missing_copy_number = missing_copies.get(protein_id, 0)
                if missing_copy_number > 0:
                    if not isinstance(model_paths, list):
                        model_paths = [model_paths]  # Ensure model_paths is a list

                    for model_idx, model_path in enumerate(model_paths):
                        if model_path is None:
                            continue
                        ensemble_id = f"{protein_id}_ensemble_{model_idx}"

                        phaser_info['ensembles'][ensemble_id] = {
                            'protein_id': protein_id,
                            'model_path': model_path,
                            'copies': missing_copies
                        }

                        f.write("  ensemble {\n")
                        f.write(f"    model_id = {ensemble_id}\n")
                        f.write("    coordinates {\n")
                        f.write(f"      pdb = {model_path}\n")
                        f.write("      identity = 90.0\n")
                        f.write("    }\n")
                        f.write("  }\n")

                        f.write("  search {\n")
                        f.write(f"    ensembles = {ensemble_id}\n")
                        f.write(f"    copies = {missing_copy_number}\n")
                        f.write("  }\n")

                        # ensemble_index += 1

            f.write("  keywords {\n")
            f.write("    general {\n")
            f.write(f"      jobs = {nproc}\n")
            f.write("    }\n")
            f.write("  }\n")
            f.write("}\n")


    def find_autobuild_inputs(self, output_root):
        latest_phaser_mtz = None
        latest_phaser_pdb = None
        latest_mod_time = 0

        # Walk through the directory to find the latest PHASER.1.mtz
        for root, dirs, files in os.walk(output_root):
            for file in files:
                if file == "PHASER.1.mtz":
                    mtz_path = os.path.join(root, file)
                    mod_time = os.path.getmtime(mtz_path)

                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        latest_phaser_mtz = mtz_path

        # If the latest PHASER.1.mtz is found, check for *_partial.pdb in the same directory
        if latest_phaser_mtz:
            latest_phaser_pdb_dir = os.path.dirname(latest_phaser_mtz)
            partial_pdbs = glob.glob(os.path.join(latest_phaser_pdb_dir, "*_partial.pdb"))

            if partial_pdbs:
                latest_phaser_pdb = partial_pdbs[0]  # Select the first *_partial.pdb if available
            else:
                # Fall back to PHASER.1.pdb if no partial pdb is found
                pdb_path = os.path.join(latest_phaser_pdb_dir, "PHASER.1.pdb")
                if os.path.exists(pdb_path):
                    latest_phaser_pdb = pdb_path

        return latest_phaser_pdb, latest_phaser_mtz