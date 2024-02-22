
import time
import os
import sys
import re
import shutil
import logging
import argparse
import glob
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from Bio.PDB import PDBParser

from SequenceManager import SequenceManager
from PDBManager import PDBManager
from MolecularReplacement import MolecularReplacement
from JobMonitor import JobMonitor
from ColabFold import ColabFold
from DataManager import DataManager
import utilities

af_cluster_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "AF_cluster.py")

def parse_args():
    parser = argparse.ArgumentParser(description="Molecular replacement for multi-chain protein complexes using protein sequences and X-ray diffraction data.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file containing protein sequences for each chain.")
    parser.add_argument("--mtz_path", type=str, required=True, help="Path to the input MTZ file containing reduced X-ray diffraction data.")
    parser.add_argument("--uniprot_ids", type=str, required=False, help="Comma-separated UniProt IDs corresponding to the order of sequences in the CSV file.")
    parser.add_argument("--copy_numbers", type=str, required=False, help="Collon-separated copy numbers corresponding to the order of sequences in the CSV file.")
    parser.add_argument("--solvent_content", type=float, required=False, help="Solvent content of the crystal (default: 0.5).")
    parser.add_argument("--output_path", type=str, required=False, help="Base directory for all output files. If not specified, uses the current directory.")
    parser.add_argument("--nproc", type=int, required=False, default=8, help="Number of processors to use (default: 8).")
    parser.add_argument("--force_af_cluster", action="store_true", help="Force running AF_cluster.py even if the output directory already exists.")
    # ... include other existing arguments as required ...
    return parser.parse_args()



def main():
    start_time = time.time()
    sequence_manager = SequenceManager()
    pdb_manager = PDBManager()
    molecular_replacement = MolecularReplacement(pdb_manager)
    colabfold = ColabFold()
    data_manager = DataManager()
    job_monitor = JobMonitor()
    
    def process_and_update_ensemble_files(protein_info, sequence_id, ensemble_type, b_factor_cutoff):
        if ensemble_type == "colabfold_interpro":
            inference_type = "cf"
            domain_recognition_method = "interpro"
        elif ensemble_type == "colabfold_pae":
            inference_type = "cf"
            domain_recognition_method = "pae"
        elif ensemble_type == "alphafold_interpro":
            inference_type = "af"
            domain_recognition_method = "interpro"
        elif ensemble_type == "alphafold_pae":
            inference_type = "af"
            domain_recognition_method = "pae"
        
        ensemble_files = protein_info[sequence_id][inference_type][f"{ensemble_type}_ensembles"]
        processed_ensemble_dir = protein_info[sequence_id]["mr_models"][f"mr_model_folder_{domain_recognition_method}_mode"]
        os.makedirs(processed_ensemble_dir, exist_ok=True)

        processed_ensemble_paths = []
        for file_path in ensemble_files:
            processed_file_path = pdb_manager.process_pdb_file(
                file_path, b_factor_cutoff, os.path.join(processed_ensemble_dir, os.path.basename(file_path))
            )
            processed_ensemble_paths.append(processed_file_path)

        protein_info[sequence_id]["mr_models"][f"mr_model_path_{domain_recognition_method}_mode"] = processed_ensemble_paths
    
    utilities.setup_custom_logger()

    args = parse_args()
    # Resolve output directory to an absolute path
    if args.output_path:
        output_root = os.path.abspath(args.output_path)
    else:
        output_root = os.path.abspath(os.getcwd())  # Current directory as default
        
    protein_info = {}

    # Read sequences and UniProt IDs from CSV
    sequences = sequence_manager.read_sequences_from_csv(args.csv_path)
    uniprot_ids = args.uniprot_ids.split(',') if args.uniprot_ids else []
    print(f"Processing {len(sequences)} sequences...")
    print(f"Sequences: {sequences}")

    for idx, (sequence_id, sequence) in enumerate(sequences):
        if uniprot_ids:
            uniprot_id = uniprot_ids[idx]
        else:
            uniprot_id = None
        output_dir = utilities.create_structure_directory(sequence_id)
        protein_info[sequence_id] = {
            "sequence": sequence,
            "uniprot_id": uniprot_id,
            "uniprot_start": None, # starting position of the input sequence relative to UniProt sequence
            "start": None, # starting position of the input sequence relative to alphafold model
            "bfactor_cutoff": 60, # default bfactor cutoff
            "output_dir": os.path.join(output_root, output_dir),
            # Initialize other fields as None or default values
            "af": {
                "alphafold_model_path": None,
                "alphafold_pae_json_path": None,
                "alphafold_pae_domains": None,
                "alphafold_pae_ensembles": None,
                "alphafold_interpro_ensembles": None,
                "alphafold_pae_domains_output_dir": os.path.join(output_root, f"alphafold_pae_domains_output_{sequence_id}"),
                "alphafold_interpro_domains_output_dir": os.path.join(output_root, f"alphafold_interpro_domains_output_{sequence_id}"),                
            },
            "cf": {
                "colabfold_model_path": None,
                "colabfold_pae_json_path": None,
                "colabfold_msa": None,
                "colabfold_pae_domains": None,
                "colabfold_adjusted_interpro_domains": None,
                "colabfold_adjusted_pae_domains": None, 
                "colabfold_pae_ensembles": None,  
                "colabfold_pae_domains_output_dir": os.path.join(output_root, f"colabfold_pae_domains_output_{sequence_id}"),
                "colabfold_interpro_ensembles": None,
                "colabfold_interpro_domains_output_dir": os.path.join(output_root, f"colabfold_interpro_domains_output_{sequence_id}"),             
            },
            "best_model_path": None,
            "mr_models": {
                "mr_model_folder_default_mode": os.path.join(output_root, "mr_models", sequence_id, "default_mode"),
                "mr_model_path_default_mode": None,
                "mr_model_folder_interpro_mode": os.path.join(output_root, "mr_models", sequence_id, "interpro_mode"),
                "mr_model_path_interpro_mode": None,
                "mr_model_folder_pae_mode": os.path.join(output_root, "mr_models", sequence_id, "pae_mode"),
                "mr_model_path_pae_mode": None,
            },
            "copy_number": None,
            "interpro_domains": None,

        }
    sequence_ids_over_50 = {seq_id for seq_id, details in protein_info.items() if len(details.get('sequence', '')) > 50}

    if len(sequences) != len(uniprot_ids):
        logging.warning("Number of sequences and UniProt IDs do not match, which may result in Interpro mode not right.")
        # sys.exit(1)

    for sequence_id, protein_data in protein_info.items():
        uniprot_id = protein_data.get("uniprot_id")
        if uniprot_id:
            logging.info(f"Processing sequence {sequence_id} with UniProt ID {uniprot_id}...")
        else:
            logging.error(f"Could not process sequence {sequence_id} because no UniProt ID was supplied.")
        output_dir = utilities.create_structure_directory(sequence_id)

        # Fetch the Uniprot ID if it's not supplied
        if not uniprot_id and len(protein_info[sequence_id]['sequence']) > 100:
            logging.warning(f"Uniprot ID not supplied for sequence {sequence_id}...")
            logging.info("Fetching Uniprot ID from sequence...")

            uniprot_id = sequence_manager.get_uniprot_id_with_timeout(protein_info[sequence_id]['sequence'])

            if uniprot_id:
                logging.info(f"Deduced Uniprot ID for sequence {sequence_id}: {uniprot_id}")
            else:
                logging.warning(f"Could not fetch Uniprot ID for sequence {sequence_id}.")

        if uniprot_id:
            try:
                interpro_domains = pdb_manager.get_prioritized_domains(uniprot_id)
                logging.info(f"Interpro domains for sequence {sequence_id}: {interpro_domains}")
                if interpro_domains: #and "entry_protein_locations" in interpro_domains:
                    protein_info[sequence_id]["interpro_domains"] = interpro_domains
                    uniprot_start, uniprot_end = sequence_manager.get_absolute_positions(protein_info[sequence_id]['sequence'], uniprot_id)
                    protein_info[sequence_id]["uniprot_start"] = uniprot_start
                    # Additional processing if required
                else:
                    logging.warning(f"No interpro domains found or data fetching failed for UniProt ID {uniprot_id}. Skipping interpro mode for sequence {sequence_id}.")
                    uniprot_start = None
                    # Handle the case where interpro domains are not available or fetching failed
                    # You might want to set a flag or make adjustments to skip interpro mode for this sequence
            except Exception as e:
                logging.error(f"Error processing sequence {sequence_id} with UniProt ID {uniprot_id}: {e}")
                # Handle error or skip to next sequence
                uniprot_start = None
        else:
            uniprot_start = None
            uniprot_end = None
        # Attempt to download AlphaFold model
        try: 
            alphafold_model_path, alphafold_pae_json_path = pdb_manager.download_alphafold_model(uniprot_id, output_dir)
            if alphafold_model_path:
                model_sequence = pdb_manager.get_sequence_from_pdb(alphafold_model_path)
                start, end = sequence_manager.align_sequences(protein_info[sequence_id]['sequence'], model_sequence)
                start = 1 if start == 0 else start
                print(f"Sequence {sequence_id}: Alignment start: {start}, end: {end}.")
                protein_info[sequence_id]["start"] = start
                truncated_model_path = pdb_manager.truncate_model(alphafold_model_path, start, end)
                alphafold_model_path = truncated_model_path
                protein_info[sequence_id]["af"]["alphafold_model_path"] = alphafold_model_path
                protein_info[sequence_id]["af"]["alphafold_pae_json_path"] = alphafold_pae_json_path
            else:
                logging.warning(f"No AlphaFold model found for sequence {sequence_id}.")
                alphafold_model_path = None
                alphafold_pae_json_path = None
                start = None
                end = None
        except Exception as e:
            logging.warning(f"Could not download AlphaFold model for sequence {sequence_id}: {e}")
            alphafold_model_path = None
            alphafold_pae_json_path = None

        # Choose and process the best model (AlphaFold or ColabFold) based on pLDDT
        best_model_path = None
        # Run ColabFold if necessary
        colabfold_model_path = None
        if not job_monitor.mark_colabfold_finished(output_dir, sequence_id):
            logging.info(f"Running ColabFold for sequence {sequence_id}...")
            # first write the sequence content to a fasta file in the output directory
            fasta_filename = os.path.join(output_dir, f"{sequence_id}.fasta")
            sequence_manager.write_custom_sequence_to_fasta(sequence_id, protein_info[sequence_id]['sequence'], fasta_filename)
            # run colabfold
            colabfold.run_colabfold(fasta_filename, output_dir)
            job_monitor.mark_colabfold_finished(output_dir, sequence_id)
            logging.info(f"ColabFold finished for sequence {sequence_id}.")
        else:
            logging.info(f"ColabFold already run for sequence {sequence_id}. Skipping...")
        colabfold_model_files = glob.glob(f"{output_dir}/{sequence_id}_relaxed_rank_001*seed_000.pdb")
        colabfold_model_path = colabfold_model_files[0] if colabfold_model_files[0] else None
        colabfold_pae_json_path = glob.glob(f"{output_dir}/{sequence_id}*predicted_aligned_error_v1.json")[0] if glob.glob(f"{output_dir}/{sequence_id}*predicted_aligned_error_v1.json")[0] else None
        protein_info[sequence_id]["cf"]["colabfold_model_path"] = colabfold_model_path
        colabfold_msa = f"{output_dir}/{sequence_id}.a3m" if f"{output_dir}/{sequence_id}.a3m" else None
        protein_info[sequence_id]["cf"]["colabfold_msa"] = colabfold_msa
        
        colabfold_marker = False
        best_model_path = None
        if colabfold_model_path:
            print(f"Sequence {sequence_id}: ColabFold model: {colabfold_model_path}")
            protein_info[sequence_id]["cf"]["colabfold_pae_json_path"] = colabfold_pae_json_path
            # renumber colabfold model
            offset = uniprot_start if uniprot_start is not None else (start if start is not None else 1)
            best_model_path = pdb_manager.renumber_colabfold_model(colabfold_model_path, offset)
            colabfold_marker = True
            
            print(f"AlphaFold model path: {alphafold_model_path}")
            print(f"AlphaFold model path: {protein_info[sequence_id]['af']['alphafold_model_path']}")

            # Choose the best model (AlphaFold or ColabFold) based on pLDDT
            if protein_info[sequence_id]["af"]['alphafold_model_path'] is not None:
                alphafold_plddt = utilities.calculate_mean_plddt(protein_info[sequence_id]['af']["alphafold_model_path"])
                print(f"Sequence {sequence_id}: AlphaFold pLDDT: {alphafold_plddt}")
                colabfold_plddt = utilities.calculate_mean_plddt(colabfold_model_path)
                print(f"Sequence {sequence_id}: ColabFold pLDDT: {colabfold_plddt}")
                if alphafold_plddt > colabfold_plddt:
                    best_model_path = protein_info[sequence_id]["af"]["alphafold_model_path"]
                    colabfold_marker = False
        else:
            logging.warning(f"No ColabFold model found for sequence {sequence_id}.")
            if protein_info[sequence_id]["af"]["alphafold_model_path"] is not None:
                best_model_path = protein_info[sequence_id]["af"]["alphafold_model_path"]
                colabfold_marker = False
            else:
                logging.warning(f"No AlphaFold model found for sequence {sequence_id}.")

        if best_model_path:
            print(f"Sequence {sequence_id}: Best model: {best_model_path}")
            protein_info[sequence_id]["best_model_path"] = best_model_path
            mean_plddt_score = utilities.calculate_mean_plddt(best_model_path)
            b_factor_cutoff = 40 if mean_plddt_score < 55 else 60
            protein_info[sequence_id]["bfactor_cutoff"] = b_factor_cutoff

            mr_model_folder_default_mode = protein_data.get("mr_models", {}).get("mr_model_folder_default_mode")
            if mr_model_folder_default_mode:
                os.makedirs(mr_model_folder_default_mode, exist_ok=True)
            else:
                # Handle the case where the path is None
                logging.error(f"Path for mr_model_folder_default_mode is None for sequence {sequence_id}")
            base_name = os.path.basename(best_model_path)         
            output_pdb_path = os.path.join(mr_model_folder_default_mode, f"{base_name}_plddt{b_factor_cutoff}.pdb")

            protein_info[sequence_id]["mr_models"]["mr_model_path_default_mode"] = pdb_manager.process_pdb_file(best_model_path, b_factor_cutoff, output_pdb_path)
            if len(protein_info[sequence_id]["sequence"]) > 100:
                if colabfold_marker == True:
                    protein_info[sequence_id]["cf"]["colabfold_adjusted_interpro_domains"] = protein_info[sequence_id]["interpro_domains"]
                    protein_info[sequence_id]["cf"]["colabfold_adjusted_pae_domains"] = pdb_manager.get_domain_definitions_from_pae(
                        protein_info[sequence_id]["cf"]["colabfold_pae_json_path"], offset
                    )
                    # truncate colabfold model with interpro domains
                    colabfold_interpro_domains_output_dir = protein_data.get("cf", {}).get("colabfold_interpro_domains_output_dir")
                    os.makedirs(colabfold_interpro_domains_output_dir, exist_ok=True)
                    protein_info[sequence_id]["cf"]["colabfold_interpro_ensembles"] = pdb_manager.prepare_domain_ensembles(
                        best_model_path, protein_info[sequence_id]["cf"]["colabfold_adjusted_interpro_domains"],
                        colabfold_interpro_domains_output_dir, len(protein_info[sequence_id]["sequence"]))
                    process_and_update_ensemble_files(protein_info, sequence_id, "colabfold_interpro", b_factor_cutoff)

                    # truncate colabfold model with pae domains
                    os.makedirs(protein_info[sequence_id]["cf"]["colabfold_pae_domains_output_dir"], exist_ok=True)
                    protein_info[sequence_id]["cf"]["colabfold_pae_ensembles"] = pdb_manager.prepare_domain_ensembles(
                        best_model_path, protein_info[sequence_id]["cf"]["colabfold_adjusted_pae_domains"],
                        protein_info[sequence_id]["cf"]["colabfold_pae_domains_output_dir"], len(protein_info[sequence_id]["sequence"]))
                    process_and_update_ensemble_files(protein_info, sequence_id, "colabfold_pae", b_factor_cutoff)

                else:
                    protein_info[sequence_id]["af"]["alphafold_pae_domains"] = pdb_manager.get_domain_definitions_from_pae(
                        protein_info[sequence_id]["af"]["alphafold_pae_json_path"]
                    )
                    # truncate alphafold model with interpro domains
                    os.makedirs(protein_info[sequence_id]["af"]["alphafold_interpro_domains_output_dir"], exist_ok=True)
                    protein_info[sequence_id]["af"]["alphafold_interpro_ensembles"] = pdb_manager.prepare_domain_ensembles(
                        best_model_path, protein_info[sequence_id]["interpro_domains"],
                        protein_info[sequence_id]["af"]["alphafold_interpro_domains_output_dir"], len(protein_info[sequence_id]["sequence"]))
                    process_and_update_ensemble_files(protein_info, sequence_id, "alphafold_interpro", b_factor_cutoff)

                    # truncate alphafold model with pae domains
                    os.makedirs(protein_info[sequence_id]["af"]["alphafold_pae_domains_output_dir"], exist_ok=True)
                    protein_info[sequence_id]["af"]["alphafold_pae_ensembles"] = pdb_manager.prepare_domain_ensembles(
                        best_model_path, protein_info[sequence_id]["af"]["alphafold_pae_domains"],
                        protein_info[sequence_id]["af"]["alphafold_pae_domains_output_dir"], len(protein_info[sequence_id]["sequence"]))
                    process_and_update_ensemble_files(protein_info, sequence_id, "alphafold_pae", b_factor_cutoff)

            else:
                logging.warning(f"Sequence {sequence_id} is too short to run interpro/pae mode. Skipping...")
        else:
            logging.warning(f"No suitable model found for sequence {sequence_id}. Skipping this sequence.")
    
    sequence_ids = list(protein_info.keys())
    if args.copy_numbers:
        copy_numbers_list = args.copy_numbers.split(':')
        # sequence_ids = list(protein_info.keys())  # Extract sequence IDs from protein_info dictionary
        if len(copy_numbers_list) != len(sequence_ids):
            logging.error("Number of specified copy numbers does not match the number of sequences in the CSV file.")
            sys.exit(1)
        specified_copy_numbers = {sequence_id: int(copy_num) for sequence_id, copy_num in zip(sequence_ids, copy_numbers_list)}
    else:
        specified_copy_numbers = None

    logging.info(f"User specified copy numbers: {specified_copy_numbers}")

    # Call the modified function
    closest_combination, all_combinations, mean_matthews_coeff = molecular_replacement.multi_analyze_asu_and_solvent_content(
        args.mtz_path, args.csv_path, sequence_ids, specified_copy_numbers
    )
    
    # Analyze ASU and find the number of copies for each sequence and solvent content
    # closest_combination, all_combinations, mean_matthews_coeff = molecular_replacement.multi_analyze_asu_and_solvent_content(args.mtz_path, args.csv_path, specified_copy_numbers)

    if closest_combination:
        favorable_combination_str, prob, solvent_content, matthews_coeff = closest_combination
        if args.solvent_content:
            solvent_content = args.solvent_content
        updated_solvent_content = solvent_content
        copy_numbers_favorable = {protein: int(copy_num) for protein, copy_num in (protein_copy_combo.split(" x ") for protein_copy_combo in favorable_combination_str.split(", "))}

        logging.info(f"Favorable combination of copy numbers: {favorable_combination_str}")
        logging.info(f"Copy numbers: {copy_numbers_favorable}")
        logging.info(f"Solvent content: {solvent_content:.2f}")
        logging.info(f"Matthews Coefficient: {matthews_coeff:.2f}")
    else:
        logging.error("Could not find a favorable combination of copy numbers. Please supply the copy numbers using the --copy_numbers argument.")
        sys.exit(1)

    space_group = data_manager.get_space_group(args.mtz_path)
    print(f"Space group: {space_group}")

    # Update the copy numbers in protein_info
    for sequence_id in protein_info:
        protein_info[sequence_id]['copy_number'] = copy_numbers_favorable.get(sequence_id, 1)

    mr_model_paths = {sequence_id: protein_info[sequence_id]["mr_models"]["mr_model_path_default_mode"] for sequence_id in protein_info}
    copy_numbers = {sequence_id: protein_info[sequence_id]["copy_number"] for sequence_id in protein_info}
    print(f"final checking copy_numbers: {copy_numbers}")

    # define a series of variables 
    phaser_info = {
        "default_mode": { 
            "success": {
                "01": False,
                "02": False,
                },
            "output_dir": {
                "01": os.path.join(output_root, "default_phaser_output01"),
                "02": os.path.join(output_root, "default_phaser_output02"),
                },
            "params_file": {
                "01": os.path.join(output_root, "default_phaser_output01", "default_phaser.params"),
                "02": os.path.join(output_root, "default_phaser_output02", "default_phaser.params"), 
                },
            "phaser_log": {
                "01": os.path.join(output_root, "default_phaser_output01", "PHASER.log"),
                "02": os.path.join(output_root, "default_phaser_output02", "PHASER.log"),
                },
            "tfz_score": 0
        },
        "interpro_mode": {
            "success": False,
            "output_dir": os.path.join(output_root, "interpro_phaser_output"),
            "params_file": os.path.join(output_root, "interpro_phaser_output", "interpro_phaser.params"),
            "tfz_score": 0,
            "interpro_switch": "off"
        },
        "pae_mode": {
            "success": False,
            "output_dir": os.path.join("pae_phaser_output"),
            "params_file": os.path.join("pae_phaser_output", "pae_phaser.params"),
            "tfz_score": 0,
            "pae_switch": "off"
        },
        "AF_cluster_mode": {
            "success": False,
            "output_dir": None,
            "tfz_score": 0,
            "af_cluster_switch": "off"
        },
        "best_tfz_score": 0,
        "mr_solution": False,
        "high_resolution": 0,
        "is_successful": False,
        "protein_id": "",
    }

    """
    Phaser default mode 1st attempt
    """
    total_copies_to_search = copy_numbers

    # reinitiate copy_numbers's value to 0 but keep the keys and call it total_found_copies
    total_found_copies = {protein_id: 0 for protein_id in copy_numbers}
    total_missing_copies = total_copies_to_search
    partial_pdb_path = None

    if os.path.exists(phaser_info["default_mode"]["output_dir"]['01']):
        logging.info("default phaser output directory already exists. default phaser mode won't be run.")
        logging.warning(f"if you want to run default phaser mode, please delete the directory {phaser_info['default_mode']['output_dir']['01']} and rerun the script.")
        # tempararily add
        molecular_replacement.generate_phaser_params_multimer(phaser_info["default_mode"]["params_file"]['01'], 
            args.mtz_path, solvent_content, space_group, mr_model_paths, 
            copy_numbers, phaser_info, nproc=args.nproc) 
    else:
        os.makedirs(phaser_info["default_mode"]["output_dir"]['01'], exist_ok=True)
        logging.info(f"running phaser molecular replacement with default mode, see parameters used in {phaser_info['default_mode']['params_file']['01']}...")
        logging.info(f"for default mode 1st attempt, copy numbers are {copy_numbers}")
 
        molecular_replacement.generate_phaser_params_multimer(phaser_info["default_mode"]["params_file"]['01'], 
            args.mtz_path, solvent_content, space_group, mr_model_paths, 
            copy_numbers, phaser_info, nproc=args.nproc)        
        molecular_replacement.run_phaser_molecular_replacement(phaser_info["default_mode"]["params_file"]['01'])
        
    phaser_info["default_mode"]["success"]['01'] = molecular_replacement.handle_phaser_output(phaser_info["default_mode"]["output_dir"]['01'])

    if phaser_info["default_mode"]['success']['01']:
        phaser_info["default_mode"]['tfz_score'], LLG = molecular_replacement.get_final_tfz(phaser_info["default_mode"]["output_dir"]['01'])
        logging.success(f"TFZ score for default phaser run: {phaser_info['default_mode']['tfz_score']}, LLG: {LLG}")
        logging.success("default phaser molecular replacement 1st attempt successful, because at least some components were found.\nAnalyzing phaser output looking for potential adjustment...")

        # Parse the Phaser log file to find out which chains to keep
        phaser_log_path = phaser_info["default_mode"]["phaser_log"]['01']  # Ensure this path is set correctly
        phaser_output_pdb = os.path.join(phaser_info["default_mode"]["output_dir"]['01'], "PHASER.1.pdb")  # Path to Phaser's output PDB
        partial_pdb_path = os.path.join(phaser_info["default_mode"]["output_dir"]['01'], "default_mode_partial.pdb")
        
        total_expected_chains = sum(copy_numbers.values())
        keep_chains = pdb_manager.parse_phaser_log(phaser_log_path)
        logging.info(f"Chains to keep for default mode 01: {keep_chains}")
        kept_chain_count = sum(keep for _, keep in keep_chains)
        print(f"Chains to keep: {keep_chains}")
        pdb_manager.process_pdb_file_for_phaser(phaser_output_pdb, keep_chains, partial_pdb_path)

        if kept_chain_count != total_expected_chains:
            logging.warning("Partial success: Not all expected chains are kept.")
            # pdb_manager.process_pdb_file_for_phaser(phaser_output_pdb, keep_chains, partial_pdb_path)

            # Deduce next possible copy numbers
            
            updated_copy_numbers, updated_solvent_content, found_copies = molecular_replacement.deduce_missing_copies(phaser_log_path, phaser_info, all_combinations, mean_matthews_coeff)
            if args.copy_numbers:
                updated_copy_numbers = copy_numbers
                updated_solvent_content = solvent_content

            # print(f"Sequence IDs with more than 50 residues: {sequence_ids_over_50}")
            all_sequences_found = all(seq_id in found_copies and found_copies[seq_id] > 0 for seq_id in sequence_ids_over_50) # a boolean indicating whether each sequence ID in sequence_ids_over_50 has at least one chain found in the first Phaser attempt.
            # print(f"status of all_sequences_found: {all_sequences_found}")
            total_found_copies = found_copies
            logging.info(f"Found copies from 1st default attempt: {found_copies}")            
            if all_sequences_found and updated_copy_numbers is not None:
                logging.info(f"All components successfully located at least once for each, according to all_sequences_found: {all_sequences_found}. Found copies: {found_copies}.")
                total_copies_to_search = updated_copy_numbers
                """
                default mode 2nd attempt
                """            
                logging.info(f"Updated copy numbers for 2nd default attempt: {updated_copy_numbers}")
                print(f"Updated solvent content: {updated_solvent_content}")

                # Compare the original and updated copy numbers, and proceed with the second Phaser run if they are different
                if copy_numbers != updated_copy_numbers or total_found_copies != updated_copy_numbers:
                    logging.warning(f"1st default phaser molecular replacement was partially successful. Preparing for a 2nd default phaser molecular replacement with updated copy numbers. {updated_copy_numbers}")

                    # Calculate missing copies for the second Phaser run
                    missing_copies = {protein_id: updated_copy_numbers[protein_id] - found_copies.get(protein_id, 0) for protein_id in updated_copy_numbers}
                    logging.info(f"Missing copies for 2nd default attempt: {missing_copies}")
                    total_missing_copies = missing_copies
                    # if total_missing_copies is none then consider the 1st attempt as successful
                    if all(value == 0 for value in total_missing_copies.values()):
                        logging.info("default phaser molecular replacement are considered successful, although there may be some components missing, further visualization of the output pdb and map is needed.")
                        phaser_info['interpro_mode']['interpro_switch'] = 'off'

                    else:
                        if os.path.exists(phaser_info["default_mode"]["output_dir"]['02']):
                            logging.info("default phaser output directory already exists. default phaser mode won't be run.")
                            logging.warning(f"if you want to run default phaser mode, please delete the directory {phaser_info['default_mode']['output_dir']['02']} and rerun the script.")
                            # tempararily add
                            molecular_replacement.generate_phaser_params_for_second_run(
                                phaser_info["default_mode"]["params_file"]['02'], args.mtz_path, updated_solvent_content, 
                                space_group, partial_pdb_path, mr_model_paths, missing_copies, phaser_info, nproc=args.nproc
                            ) 
                        else:
                            # Prepare for the second Phaser run
                            os.makedirs(phaser_info["default_mode"]["output_dir"]['02'], exist_ok=True)
            
                            molecular_replacement.generate_phaser_params_for_second_run(
                                phaser_info["default_mode"]["params_file"]['02'], args.mtz_path, updated_solvent_content, 
                                space_group, partial_pdb_path, mr_model_paths, missing_copies, phaser_info, nproc=args.nproc
                            ) 
                            molecular_replacement.run_phaser_molecular_replacement(phaser_info["default_mode"]["params_file"]['02'])
                            
                        phaser_info["default_mode"]['success']['02'] = molecular_replacement.handle_phaser_output(phaser_info["default_mode"]["output_dir"]['02'])
                        default_2nd_output_pdb = os.path.join(phaser_info["default_mode"]["output_dir"]['02'], "PHASER.1.pdb")  # Path to Phaser's output PDB
                        if phaser_info["default_mode"]['success']['02']:
                            phaser_info["default_mode"]['tfz_score'], LLG = molecular_replacement.get_final_tfz(phaser_info["default_mode"]["output_dir"]['02'])
                            logging.success(f"TFZ score for default phaser run round 2: {phaser_info['default_mode']['tfz_score']}, LLG: {LLG}")
                            logging.success("default phaser molecular replacement 2nd attempt successful, because all the rest components were found.\nAnalyzing phaser output looking for potential adjustment...")

                            # Check if the newly found copies match the missing copies from the first attempt
                            _, _, new_found_copies = molecular_replacement.deduce_missing_copies(phaser_info["default_mode"]["phaser_log"]['02'], phaser_info, all_combinations, mean_matthews_coeff)
                            logging.info(f"Newly found copies during 2nd phaser attemp: {new_found_copies}")
                            total_found_copies = {protein_id: total_found_copies.get(protein_id, 0) + new_found_copies.get(protein_id, 0) for protein_id in total_found_copies}
                            logging.info(f"Total found copies after 2nd default mode: {total_found_copies}")
                            if new_found_copies == missing_copies:
                                logging.success(f"All components successfully located: {total_found_copies}. No further searches needed.")
                                phaser_info["default_mode"]["success"]['02'] = True
                            else:
                                logging.warning("Not all components located. Preparing for Interpro mode.")
                                # Update copy numbers for Interpro mode
                                interpro_copy_numbers = {protein_id: missing_copies.get(protein_id, 0) - new_found_copies.get(protein_id, 0) for protein_id in missing_copies}
                                print(f"Interpro copy numbers not equals missing_copies: {interpro_copy_numbers}")
                                total_missing_copies = interpro_copy_numbers
                                phaser_info['interpro_mode']['interpro_switch'] = 'on' # switch to interpro mode
                            # Generate new partial pdb for Interpro mode if necessary
                            keep_chains = pdb_manager.parse_phaser_log(phaser_info["default_mode"]["phaser_log"]['02'])
                            default_2nd_partial_pdb_path = os.path.join(phaser_info["default_mode"]["output_dir"]['02'], "default_mode_2nd_partial.pdb")
                            pdb_manager.process_pdb_file_for_phaser(default_2nd_output_pdb, keep_chains, default_2nd_partial_pdb_path, partial_pdb_path)
                            # update partial_pdb_path
                            partial_pdb_path = default_2nd_partial_pdb_path
                        else:
                            logging.warning("2nd default phaser molecular replacement failed. Will carry on with interpro/pae mode.")
                            phaser_info['interpro_mode']['interpro_switch'] = 'on' 
                            # Use missing_copies from the 1st attempt for the Interpro mode
                            interpro_copy_numbers = missing_copies
                            print(f"Interpro copy numbers equals missing_copies : {interpro_copy_numbers}")
                            # Use the same partial pdb as in the 2nd attempt
            else:
                total_copies_to_search = copy_numbers
                total_missing_copies = {protein_id: total_copies_to_search.get(protein_id, 0) - total_found_copies.get(protein_id, 0) for protein_id in total_copies_to_search}
                logging.warning(f"some components were found, but not all: all_sequences_found {all_sequences_found}. Preparing for Interpro mode.")
                phaser_info['interpro_mode']['interpro_switch'] = 'on'
                phaser_info["best_tfz_score"] = phaser_info["default_mode"]['tfz_score']
                phaser_info["mr_solution"] = True        
        else:
            logging.success("default phaser molecular replacement successful, because all components were found.")
            phaser_info["default_mode"]["success"]['01'] = True
            phaser_info['interpro_mode']['interpro_switch'] = 'off'
    else:
        logging.warning("default phaser molecular replacement failed.")
        phaser_info['interpro_mode']['interpro_switch'] = 'on'

    if phaser_info['interpro_mode']['interpro_switch'] == 'off': # update best_tfz
        phaser_info["best_tfz_score"] = phaser_info["default_mode"]['tfz_score']
        phaser_info["mr_solution"] = True
    """
    Run Phaser with interpro mode
    """
    if phaser_info['interpro_mode']['interpro_switch'] == 'on':
        # Set up the paths and parameters for Interpro mode
        interpro_mode_dir = os.path.join(output_root, "interpro_phaser_output")
        interpro_params_file = os.path.join(interpro_mode_dir, "interpro_phaser.params")
        interpro_log_file = os.path.join(interpro_mode_dir, "PHASER.log")
        interpro_copy_numbers = total_missing_copies
        print(f"Interpro copy numbers which is the same as total_missing_copies: {interpro_copy_numbers}")
        logging.info(f"Running phaser molecular replacement with interpro mode, see parameters used in {interpro_params_file}...")
        logging.info(f"for interpro mode, copy numbers to search for are {interpro_copy_numbers}")
        if os.path.exists(interpro_mode_dir):
            logging.info("interpro phaser output directory already exists. interpro phaser mode won't be run.")
            logging.warning(f"if you want to run interpro phaser mode, please delete the directory {interpro_mode_dir} and rerun the script.")
            """
            this part is tempararily added
            """
            # Determine the copy numbers and MR models for Interpro mode
            if not phaser_info["default_mode"]['success']['01']:
                # If 1st default mode failed, update MR models and copy numbers for Interpro truncated models
                interpro_mr_models = {sequence_id: protein_info[sequence_id]["mr_models"]["mr_model_path_interpro_mode"] for sequence_id in protein_info}
                # interpro_copy_numbers = {sequence_id: protein_info[sequence_id]["copy_number"] for sequence_id in protein_info}
                # Generate and run Phaser parameters for Interpro mode
                molecular_replacement.generate_phaser_params_multimer(
                    interpro_params_file, args.mtz_path, solvent_content, 
                    space_group, interpro_mr_models, interpro_copy_numbers, phaser_info, nproc=args.nproc
                )
            else:
                # If 2nd default mode failed, use missing_copies and the same partial PDB
                # interpro_partial_pdb_path = partial_pdb_path  # Use the same partial PDB from the default mode
                interpro_mr_models = {protein_id: protein_info[protein_id]["mr_models"]["mr_model_path_interpro_mode"] for protein_id in total_missing_copies}
                # Generate and run Phaser parameters for Interpro mode
                molecular_replacement.generate_phaser_params_for_second_run(
                    interpro_params_file, args.mtz_path, updated_solvent_content,
                    space_group, partial_pdb_path, interpro_mr_models, interpro_copy_numbers, phaser_info, nproc=args.nproc
                )

        else:
            os.makedirs(interpro_mode_dir, exist_ok=True)

            # Determine the copy numbers and MR models for Interpro mode
            if not phaser_info["default_mode"]['success']['01']:
                # If 1st default mode failed, update MR models and copy numbers for Interpro truncated models
                interpro_mr_models = {sequence_id: protein_info[sequence_id]["mr_models"]["mr_model_path_interpro_mode"] for sequence_id in protein_info}
                # interpro_copy_numbers = {sequence_id: protein_info[sequence_id]["copy_number"] for sequence_id in protein_info}
                # Generate and run Phaser parameters for Interpro mode
                molecular_replacement.generate_phaser_params_multimer(
                    interpro_params_file, args.mtz_path, solvent_content, 
                    space_group, interpro_mr_models, interpro_copy_numbers, phaser_info, nproc=args.nproc
                )
            else:
                # If 2nd default mode failed, use missing_copies and the same partial PDB
                # interpro_partial_pdb_path = partial_pdb_path  # Use the same partial PDB from the default mode
                interpro_mr_models = {protein_id: protein_info[protein_id]["mr_models"]["mr_model_path_interpro_mode"] for protein_id in total_missing_copies}
                # Generate and run Phaser parameters for Interpro mode
                molecular_replacement.generate_phaser_params_for_second_run(
                    interpro_params_file, args.mtz_path, updated_solvent_content,
                    space_group, partial_pdb_path, interpro_mr_models, interpro_copy_numbers, phaser_info, nproc=args.nproc
                )

            molecular_replacement.run_phaser_molecular_replacement(interpro_params_file)

        # Handle the output of the Interpro mode
        phaser_info["interpro_mode"]['success'] = molecular_replacement.handle_phaser_output(interpro_mode_dir)
        if phaser_info["interpro_mode"]['success']:
            phaser_info["interpro_mode"]['tfz_score'], LLG = molecular_replacement.get_final_tfz(interpro_mode_dir)
            logging.success(f"TFZ score for interpro phaser run: {phaser_info['interpro_mode']['tfz_score']}, LLG: {LLG}")
            logging.success("Interpro phaser molecular replacement successful.")
            logging.info(f"After interpro mode, all combinations of copy numbers are {all_combinations}")
            logging.info(f"phaser_info is {phaser_info}")
            _, _, new_found_copies = molecular_replacement.deduce_missing_copies(interpro_log_file, phaser_info, all_combinations, mean_matthews_coeff, top_switch=False)
            logging.info(f"Newly found copies during interpro phaser attempt: {new_found_copies}")
            total_found_copies = {protein_id: total_found_copies.get(protein_id, 0) + new_found_copies.get(protein_id, 0) for protein_id in total_found_copies}
            logging.info(f"Total found copies after interpro mode: {total_found_copies}")
            total_missing_copies = {protein_id: total_missing_copies.get(protein_id, 0) - new_found_copies.get(protein_id, 0) for protein_id in total_missing_copies}
            if new_found_copies == interpro_copy_numbers:
                logging.success("All components successfully located. No further searches needed.")
                phaser_info["interpro_mode"]['success'] = True
            else:
                logging.warning("Not all components located. Preparing for pae mode.")
                phaser_info['pae_mode']['pae_switch'] = 'on'
            # Generate new partial pdb for pae mode if necessary
            keep_chains = pdb_manager.parse_phaser_log(interpro_log_file)
            interpro_phaser_output_pdb = os.path.join(interpro_mode_dir, "PHASER.1.pdb")  # Path to Phaser's output PDB
            interpro_partial_pdb_path = os.path.join(interpro_mode_dir, "interpro_partial.pdb")
            pdb_manager.process_pdb_file_for_phaser(interpro_phaser_output_pdb, keep_chains, interpro_partial_pdb_path, partial_pdb_path)
            # update partial_pdb_path
            partial_pdb_path = interpro_partial_pdb_path
        else:
            logging.warning("Interpro phaser molecular replacement failed.")
            # proceed with pae mode
            phaser_info['pae_mode']['pae_switch'] = 'on'
            # no need to update the copies
            # since the interpro mode failed, use the same partial pdb as in the interpro mode
            
        # Update the best TFZ score and MR solution status
        if phaser_info["interpro_mode"]['tfz_score'] > phaser_info["best_tfz_score"]:
            phaser_info["best_tfz_score"] = phaser_info["interpro_mode"]['tfz_score']
            phaser_info["mr_solution"] = True

    """
    Run Phaser with pae mode
    """
    if phaser_info['pae_mode']['pae_switch'] == 'on':
        pae_mode_dir = os.path.join(output_root, "pae_phaser_output")
        pae_params_file = os.path.join(pae_mode_dir, "pae_phaser.params")
        pae_log_file = os.path.join(pae_mode_dir, "PHASER.log")
        pae_copy_numbers = total_missing_copies
        print(f"pae copy numbers which is the same as total_missing_copies: {pae_copy_numbers}")
        if os.path.exists(pae_mode_dir):
            logging.info("pae phaser output directory already exists. pae phaser mode won't be run.")
            logging.warning(f"if you want to run pae phaser mode, please delete the directory {pae_mode_dir} and rerun the script.")
        else:
            os.makedirs(pae_mode_dir, exist_ok=True)
            logging.info(f"Running phaser molecular replacement with pae mode, see parameters used in {pae_params_file}...")
            logging.info(f"for pae mode, copy numbers to search for are {pae_copy_numbers}")
            if updated_solvent_content is None:
                updated_solvent_content = solvent_content
            if not phaser_info["interpro_mode"]['success']:
                # If interpro mode failed, update MR models and copy numbers for pae truncated models
                pae_mr_models = {sequence_id: protein_info[sequence_id]["mr_models"]["mr_model_path_pae_mode"] for sequence_id in protein_info}
                # pae_copy_numbers = {sequence_id: protein_info[sequence_id]["copy_number"] for sequence_id in protein_info}
                # Generate and run Phaser parameters for pae mode
                if partial_pdb_path is None:
                    molecular_replacement.generate_phaser_params_multimer(
                        pae_params_file, args.mtz_path, solvent_content, 
                        space_group, pae_mr_models, pae_copy_numbers, phaser_info, nproc=args.nproc
                    )
                else:
                    molecular_replacement.generate_phaser_params_for_second_run(
                        pae_params_file, args.mtz_path, updated_solvent_content,
                        space_group, partial_pdb_path, pae_mr_models, pae_copy_numbers, phaser_info, nproc=args.nproc
                    )
            else:
                # If interpro mode succeeded, use missing_copies and inherit partial PDB after interpro mode
                pae_partial_pdb_path = partial_pdb_path
                pae_mr_models = {protein_id: protein_info[protein_id]["mr_models"]["mr_model_path_pae_mode"] for protein_id in total_missing_copies}
                # Generate and run Phaser parameters for pae mode
                molecular_replacement.generate_phaser_params_for_second_run(
                    pae_params_file, args.mtz_path, updated_solvent_content,
                    space_group, pae_partial_pdb_path, pae_mr_models, pae_copy_numbers, phaser_info, nproc=args.nproc
                )
            molecular_replacement.run_phaser_molecular_replacement(pae_params_file)

        # Handle the output of the pae mode
        phaser_info["pae_mode"]['success'] = molecular_replacement.handle_phaser_output(pae_mode_dir)
        if phaser_info["pae_mode"]['success']:
            phaser_info["pae_mode"]['tfz_score'], LLG = molecular_replacement.get_final_tfz(pae_mode_dir)
            logging.success(f"TFZ score for pae phaser run: {phaser_info['pae_mode']['tfz_score']}, LLG: {LLG}")
            logging.success("pae phaser molecular replacement successful.")
            _, _, new_found_copies = molecular_replacement.deduce_missing_copies(pae_log_file, phaser_info, all_combinations, mean_matthews_coeff, top_switch=False)
            print(f"Newly found copies during pae phaser attemp: {new_found_copies}")
            total_found_copies = {protein_id: total_found_copies.get(protein_id, 0) + new_found_copies.get(protein_id, 0) for protein_id in total_found_copies}
            print(f"Total found copies after pae mode: {total_found_copies}")
            total_missing_copies = {protein_id: total_missing_copies.get(protein_id, 0) - new_found_copies.get(protein_id, 0) for protein_id in total_missing_copies}
            if new_found_copies == pae_copy_numbers:
                logging.success("All components successfully located. No further searches needed.")
                phaser_info["pae_mode"]['success'] = True
            else:
                logging.warning("Not all components located. Preparing for AF_cluster mode.")
                phaser_info['AF_cluster_mode']['af_cluster_switch'] = 'on'
            # Generate new partial pdb for AF_cluster mode if necessary
            keep_chains = pdb_manager.parse_phaser_log(pae_log_file)
            pae_phaser_output_pdb = os.path.join(pae_mode_dir, "PHASER.1.pdb")  # Path to Phaser's output PDB
            pae_partial_pdb_path = os.path.join(pae_mode_dir, "pae_partial.pdb")
            pdb_manager.process_pdb_file_for_phaser(pae_phaser_output_pdb, keep_chains, pae_partial_pdb_path, partial_pdb_path)
            # update partial_pdb_path
            partial_pdb_path = pae_partial_pdb_path
        else:
            logging.warning("pae phaser molecular replacement failed.")
            # proceed with AF_cluster mode
            phaser_info['AF_cluster_mode']['af_cluster_switch'] = 'on'
            # no need to update the copies
            # since the pae mode failed, use the same partial pdb as in the pae mode

    """
    AF_cluster mode
    """

    if phaser_info['AF_cluster_mode']['af_cluster_switch'] == 'on':
        # Set up the paths and parameters for AF_cluster mode
        AF_cluster_mode_root = os.path.join(output_root, "AF_cluster_root")
        if os.path.exists(AF_cluster_mode_root):
            logging.info("AF_cluster mode output directory already exists. AF_cluster mode won't be run.")
            logging.warning(f"if you want to run AF_cluster mode, please delete the directory {AF_cluster_mode_root} and rerun the script.")
        else:
            # for the protein in total_missing_copies, prepare the AF_cluster_mode_dir
            logging.info(f"total_missing_copies right before AF_cluster mode: {total_missing_copies}")
            logging.info(f"total_found_copies right before AF_cluster mode: {total_found_copies}")
            logging.info(f"proteins with more than 50 residues: {sequence_ids_over_50}")
            # Ensure no negative values in total_missing_copies
            for protein_id in total_missing_copies:
                if total_missing_copies[protein_id] < 0:
                    total_missing_copies[protein_id] = 0
            # Find proteins that are both missing (with count > 0) and have more than 50 residues
            proteins_for_AF_cluster = {protein_id for protein_id in total_missing_copies if total_missing_copies[protein_id] > 0 and protein_id in sequence_ids_over_50}

            if args.force_af_cluster:
                logging.warning("force_af_cluster is set to True. ")
                proteins_for_AF_cluster = {protein_id for protein_id in total_missing_copies if total_missing_copies[protein_id] > 0}
            sequence_lengths = {protein_id: len(protein_info[protein_id]["sequence"]) for protein_id in protein_info}
            sorted_proteins_for_AF_cluster = sorted(proteins_for_AF_cluster, key=lambda protein_id: sequence_lengths[protein_id], reverse=True)            
            
            for protein_id in sorted_proteins_for_AF_cluster:
                AF_cluster_mode_dir = os.path.join(AF_cluster_mode_root, protein_id) 
                os.makedirs(AF_cluster_mode_dir, exist_ok=True)                       
                af_cluster_reference_pdb_path = protein_info[protein_id]["mr_models"]["mr_model_path_default_mode"]
                logging.info(f"Calling run_af_cluster with arguments: {protein_info[protein_id]['cf']['colabfold_msa']}, {af_cluster_reference_pdb_path}")
                af_cluster_process = subprocess.Popen([sys.executable, af_cluster_script_path, protein_info[protein_id]['cf']['colabfold_msa'], af_cluster_reference_pdb_path, str(40)], cwd=AF_cluster_mode_dir)

                AF_cluster_phaser_output_dir = phaser_info['AF_cluster_mode']['output_dir']

                predictions_folder = os.path.join(AF_cluster_mode_dir, "predictions")
                selectives_folder = os.path.join(AF_cluster_mode_dir, "AF_cluster_selectives")
                af_cluster_finish_file = os.path.join(selectives_folder, "ALL_AFCLUSTER_DONE")
                rmsd_ranking_file = os.path.join(selectives_folder, "rmsd_ranking.csv")
                AF_cluster_treated_pdb_dir = os.path.join(AF_cluster_mode_dir, "AF_cluster_plddt_treated_pdbs")
                af_cluster_phaser_base_dir = os.path.join(AF_cluster_mode_dir, "AF_cluster_phaser_runs")
                os.makedirs(af_cluster_phaser_base_dir, exist_ok=True)
                AF_cluster_success = False

                used_pdbs = set()
                while not AF_cluster_success:
                    if os.path.exists(rmsd_ranking_file):
                        next_pdb = pdb_manager.get_next_pdb_entry(rmsd_ranking_file, used_pdbs)
                        if next_pdb is None and os.path.exists(af_cluster_finish_file):
                            logging.warning("All PDB entries from rmsd_ranking.csv have been tested with no success.")
                            break
                        while next_pdb is None and not os.path.exists(af_cluster_finish_file):
                            time.sleep(30)
                            next_pdb = pdb_manager.get_next_pdb_entry(rmsd_ranking_file, used_pdbs)
                        if next_pdb is not None:
                            used_pdbs.add(next_pdb)
                            # cluster_pdb_paths = []
                            cluster_cf_pdb_path = os.path.join(selectives_folder, next_pdb)
                            print(f"next pdb to be processed: {cluster_cf_pdb_path}")
                            cluster_number = re.search("cluster_(\d+)_", cluster_cf_pdb_path).group(1)
                            cluster_cf_pae_path = os.path.join(selectives_folder, f"cluster_{cluster_number}_predicted_aligned_error_v1.json")
                            # b_factor_cutoff = protein_info[protein_id]["bfactor_cutoff"]
                            processed_pdb_path = f"{AF_cluster_treated_pdb_dir}/cluster_{cluster_number}_plddt{b_factor_cutoff}.pdb"
                            uniprot_start = protein_info[protein_id]["uniprot_start"]
                            start = protein_info[protein_id]["start"]
                            offset = uniprot_start if uniprot_start is not None else (start if start is not None else 1)
                            best_model_path = pdb_manager.renumber_colabfold_model(cluster_cf_pdb_path, offset)
                            mean_plddt_score = utilities.calculate_mean_plddt(best_model_path)
                            b_factor_cutoff = 40 if mean_plddt_score < 55 else 60
                            logging.info(f"mean_plddt_score for {protein_id} cluster {cluster_number} is {int(mean_plddt_score)}, thus b_factor_cutoff is {b_factor_cutoff}") 
                            # Generate and run Phaser parameters for AF_cluster mode
                            mr_cluster_dir = os.path.join(af_cluster_phaser_base_dir, f"mr_{cluster_number}")
                            os.makedirs(mr_cluster_dir, exist_ok=True)
                            output_pdb_path = f"{mr_cluster_dir}/cluster_{cluster_number}_plddt{b_factor_cutoff}.pdb"
                            
                            cluster_mr_model_path = pdb_manager.process_pdb_file(best_model_path, b_factor_cutoff, output_pdb_path)
                            cluster_mr_model = {protein_id: cluster_mr_model_path}
                            params_filename = os.path.join(mr_cluster_dir, f"phaser_params_cluster_{cluster_number}.txt")
                            if partial_pdb_path is None:
                                molecular_replacement.generate_phaser_params_multimer(
                                    params_filename, args.mtz_path, solvent_content, 
                                    space_group, cluster_mr_model, {protein_id: total_missing_copies[protein_id]}, phaser_info, nproc=args.nproc
                                )
                            else:
                                molecular_replacement.generate_phaser_params_for_second_run(
                                    params_filename, args.mtz_path, solvent_content,
                                    space_group, partial_pdb_path, cluster_mr_model, {protein_id: total_missing_copies[protein_id]}, phaser_info, nproc=args.nproc
                                )
                            
                            logging.info(f"phaser params file for cluster {cluster_number} is generated and run.")
                            phaser_process = molecular_replacement.run_phaser_molecular_replacement_async(params_filename)
                            phaser_log_path = f"PHASER.log"
                            while phaser_process.poll() is None:
                                if os.path.exists(phaser_log_path):
                                    log_size_mb = os.path.getsize(phaser_log_path) / (1024 * 1024)
                                    if log_size_mb > 5:
                                        logging.warning("Terminating phaser run due to an abnormally long time.")
                                        phaser_process.terminate()
                                        break
                                time.sleep(20)  # Adjust the sleep time as needed

                            AF_cluster_mr_success = molecular_replacement.handle_phaser_output(mr_cluster_dir)
                            AF_cluster_tmp_tfz_score, LLG = molecular_replacement.get_final_tfz(mr_cluster_dir)
                            # implementation: use the renumbered_output_pdb_path, if available,  otherwise the copied processed_pdb_path in mr_cluster_dir will be used to
                            # do pae mode on that model. reference: pae_mr_models = {sequence_id: protein_info[sequence_id]["mr_models"]["mr_model_path_pae_mode"] for sequence_id in protein_info}
                            if AF_cluster_mr_success:
                                AF_cluster_tfz_score, LLG = molecular_replacement.get_final_tfz(mr_cluster_dir)
                                logging.success(f"TFZ score for AF_cluster phaser run {cluster_number}: {AF_cluster_tfz_score}, LLG: {LLG}")

                                AF_cluster_phaser_output_dir = mr_cluster_dir
                                
                                _, _, new_found_copies = molecular_replacement.deduce_missing_copies(os.path.join(AF_cluster_phaser_output_dir, "PHASER.log"), phaser_info, all_combinations, mean_matthews_coeff, top_switch=False)
                                logging.info(f"Newly found copies during AF_cluster phaser attempt: {new_found_copies}")
                                total_found_copies = {protein_id: total_found_copies.get(protein_id, 0) + new_found_copies.get(protein_id, 0) for protein_id in total_found_copies}
                                logging.info(f"Total found copies after AF_cluster mode for {protein_id}: {total_found_copies}")
                                total_missing_copies = {protein_id: total_missing_copies.get(protein_id, 0) - new_found_copies.get(protein_id, 0) for protein_id in total_missing_copies}
                                logging.info(f"Total missing copies after AF_cluster mode: {total_missing_copies}")
                                keep_chains = pdb_manager.parse_phaser_log(os.path.join(AF_cluster_phaser_output_dir, "PHASER.log"))
                                AF_cluster_phaser_output_pdb = os.path.join(AF_cluster_phaser_output_dir, "PHASER.1.pdb")  # Path to Phaser's output PDB
                                AF_cluster_partial_pdb_path = os.path.join(AF_cluster_phaser_output_dir, "AF_cluster_partial.pdb")
                                pdb_manager.process_pdb_file_for_phaser(AF_cluster_phaser_output_pdb, keep_chains, AF_cluster_partial_pdb_path, partial_pdb_path)
                                # update partial_pdb_path
                                partial_pdb_path = AF_cluster_partial_pdb_path
                                logging.info(f"***partial_pdb_path updated as: {partial_pdb_path}***")
                                logging.info(f"Current missing copy for {protein_id}: {total_missing_copies[protein_id]}")
                                if total_missing_copies[protein_id] <= 0:
                                    logging.info(f"Thus the AF_cluster for {protein_id} will be stopped.")
                                    halt_file_path = os.path.join(predictions_folder, 'HALT')
                                    with open(halt_file_path, 'w') as f:
                                        f.write("HALT")
                                    af_cluster_process.terminate
                                    AF_cluster_success = True
                                # break
                            # elif not AF_cluster_mr_success and AF_cluster_tmp_tfz_score > 6:
                            else:
                                logging.info(f"AF_cluster phaser run {cluster_number} failed, will try PAE mode for {protein_id}.")
                                
                                mr_cluster_enemble_dir = os.path.join(mr_cluster_dir, "mr_cluster_ensemble")
                                cf_adjusted_pae_domains = pdb_manager.get_domain_definitions_from_pae(cluster_cf_pae_path, offset, primary_pae_cutoff=15)
                                logging.info(f"cf_adjusted_pae_domains for {cluster_number}: {cf_adjusted_pae_domains}")
                                cluster_ensemble_files = pdb_manager.prepare_domain_ensembles(
                                    best_model_path,
                                    cf_adjusted_pae_domains, mr_cluster_enemble_dir, len(protein_info[protein_id]["sequence"])
                                )
                                logging.info(f"cluster_ensemble_files: {cluster_ensemble_files}")
                                processed_ensemble_dir = os.path.join(mr_cluster_enemble_dir, "processed_ensembles_for_pae_mr")
                                os.makedirs(processed_ensemble_dir, exist_ok=True)
                                processed_ensemble_paths = {protein_id: []}
                                for ensemble_file in cluster_ensemble_files:
                                    processed_ensemble_path = os.path.join(processed_ensemble_dir, os.path.basename(ensemble_file))
                                    pdb_manager.process_pdb_file(ensemble_file, b_factor_cutoff, processed_ensemble_path)
                                    processed_ensemble_paths[protein_id].append(processed_ensemble_path)
                                params_filename = os.path.join(mr_cluster_enemble_dir, f"phaser_params_cluster_{cluster_number}_pae_mode.txt")
                                if partial_pdb_path is None:
                                    molecular_replacement.generate_phaser_params_multimer(
                                        params_filename, args.mtz_path, solvent_content, 
                                        space_group, processed_ensemble_paths, {protein_id: total_missing_copies[protein_id]}, phaser_info, nproc=args.nproc
                                    )
                                else:
                                    molecular_replacement.generate_phaser_params_for_second_run(
                                        params_filename, args.mtz_path, solvent_content,
                                        space_group, partial_pdb_path, processed_ensemble_paths, {protein_id: total_missing_copies[protein_id]}, phaser_info, nproc=args.nproc
                                    )
                                logging.info(f"phaser params file for cluster {cluster_number} in pae mode is generated and run.")
                                phaser_process = molecular_replacement.run_phaser_molecular_replacement_async(params_filename)
                                phaser_log_path = f"PHASER.log"
                                while phaser_process.poll() is None:
                                    if os.path.exists(phaser_log_path):
                                        log_size_mb = os.path.getsize(phaser_log_path) / (1024 * 1024)
                                        if log_size_mb > 5:
                                            logging.warning("Terminating phaser run due to an abnormally long time.")
                                            phaser_process.terminate()
                                            break
                                    time.sleep(20)
                                AF_cluster_pae_success = molecular_replacement.handle_phaser_output(mr_cluster_enemble_dir)
                                if AF_cluster_pae_success:
                                    AF_cluster_tfz_score, LLG = molecular_replacement.get_final_tfz(mr_cluster_enemble_dir)
                                    logging.success(f"TFZ score for AF_cluster phaser run {cluster_number} in pae mode: {AF_cluster_tfz_score}, LLG: {LLG}")
                                    
                                    AF_cluster_phaser_output_dir = mr_cluster_enemble_dir
                                    _, _, new_found_copies = molecular_replacement.deduce_missing_copies(os.path.join(AF_cluster_phaser_output_dir, "PHASER.log"), phaser_info, all_combinations, mean_matthews_coeff, top_switch=False)
                                    logging.info(f"Newly found copies during AF_cluster phaser attempt: {new_found_copies}")
                                    total_found_copies = {protein_id: total_found_copies.get(protein_id, 0) + new_found_copies.get(protein_id, 0) for protein_id in total_found_copies}
                                    logging.info(f"Total found copies after AF_cluster mode for {protein_id}: {total_found_copies}")
                                    total_missing_copies = {protein_id: total_missing_copies.get(protein_id, 0) - new_found_copies.get(protein_id, 0) for protein_id in total_missing_copies}
                                    logging.info(f"Total missing copies after AF_cluster mode: {total_missing_copies}")
                                    keep_chains = pdb_manager.parse_phaser_log(os.path.join(AF_cluster_phaser_output_dir, "PHASER.log"))
                                    AF_cluster_phaser_output_pdb = os.path.join(AF_cluster_phaser_output_dir, "PHASER.1.pdb")
                                    AF_cluster_partial_pdb_path = os.path.join(AF_cluster_phaser_output_dir, "AF_cluster_partial.pdb")
                                    pdb_manager.process_pdb_file_for_phaser(AF_cluster_phaser_output_pdb, keep_chains, AF_cluster_partial_pdb_path, partial_pdb_path)
                                    # update partial_pdb_path
                                    partial_pdb_path = AF_cluster_partial_pdb_path
                                    logging.info(f"***partial_pdb_path updated as: {partial_pdb_path}***")
                                    logging.info(f"Current missing copy for {protein_id}: {total_missing_copies[protein_id]}")
                                    if total_missing_copies[protein_id] <= 0:
                                        logging.info(f"Thus the AF_cluster for {protein_id} will be stopped.")
                                        halt_file_path = os.path.join(predictions_folder, 'HALT')
                                        with open(halt_file_path, 'w') as f:
                                            f.write("HALT")
                                        af_cluster_process.terminate
                                        AF_cluster_success = True    
                                else:
                                    logging.warning(f"AF_cluster phaser run {cluster_number} failed.")
                                    AF_cluster_tfz_score = None
                    if AF_cluster_success:
                        phaser_info["AF_cluster_mode"]['success'] = True
                        break
                    if os.path.exists(f"{predictions_folder}/HALT"):
                        logging.warning("AF_cluster mode halted.")
                        af_cluster_process.terminate()
                        break

                    time.sleep(100)

                # check if all proteins in total_missing_copies have been processed; if so, break the for loop, if not, continue
                if all(count == 0 for count in total_missing_copies.values()):
                    logging.success(f"All components successfully located. Total missing copies: {total_missing_copies}.No further searches needed. Total found copies: {total_found_copies}")
                    phaser_info["AF_cluster_mode"]['success'] = True
                    break
                else:
                    logging.info(f"Final update: Total missing copies: {total_missing_copies}. Total found copies: {total_found_copies}")



    logging.info("Auto processing [Molecular Replacement] part is done. You can check the log file for details at automated_structure_solvation.log.")

    run_time = time.time() - start_time

    global stop_monitoring
    stop_monitoring = True
    successful_phaser = None
    successful_phaser_output_dir = None
    if phaser_info["default_mode"]["success"]['01'] and not phaser_info["default_mode"]["success"]['02']:
        successful_phaser = "default_mode_1st_attempt"
        successful_phaser_output_dir = phaser_info["default_mode"]["output_dir"]['01']
    elif phaser_info["default_mode"]["success"]['02']:
        successful_phaser = "default_mode_2nd_attempt"
        successful_phaser_output_dir = phaser_info["default_mode"]["output_dir"]['02']
    else:
        # Check for success in other modes
        for mode, info in phaser_info.items():
            if mode != "default_mode" and isinstance(info, dict):
                # Check if 'success' is a dictionary
                if isinstance(info.get("success"), dict):
                    if any(info["success"].values()):
                        successful_phaser = mode
                        successful_phaser_output_dir = info["output_dir"]
                        break
                else:
                    # Handle boolean 'success'
                    if info.get("success", False):
                        successful_phaser = mode
                        successful_phaser_output_dir = info["output_dir"]
                        break

    tfz_score = phaser_info["best_tfz_score"]
    resolution = data_manager.get_high_resolution(args.mtz_path)
    mr_rosetta_success = phaser_info["is_successful"] if phaser_info["mr_solution"] and phaser_info["high_resolution"] <= 2.5 else None
    # number of sequences
    num_sequences = len(sequences)
    # total sequence length added up from all sequences
    sequence_length = sum([len(sequence) for sequence_id, sequence in sequences])

    utilities.save_csv_report(
        "report.csv",
        num_sequences, 
        sequence_length, 
        run_time, 
        resolution, 
        tfz_score, 
        successful_phaser, 
        successful_phaser_output_dir, 
        mr_rosetta_success
    )
    
    autobuild_input_model, successful_phaser_map = molecular_replacement.find_autobuild_inputs(output_root)
    data_path = os.path.abspath(args.mtz_path)

    # if the resolution is better than 3.5, then use phenix.autobuild to build the model
    if resolution < 3.5:
        logging.info("The resolution is better than 3.5, will use phenix.autobuild to build the model.")
        
        # autobuild_input_model = partial_pdb_path if any(total_missing_copies[protein_id] > 0 for protein_id in sequence_ids_over_50) else successful_phaser_output_pdb
        logging.info(f"autobuild_input_model: {autobuild_input_model}")
        logging.info(f"successful_phaser_map: {successful_phaser_map}")
        master_fasta_filename = os.path.join(output_root, "master.fasta")

        with open(master_fasta_filename, "w") as f:
            for sequence_id, sequence in sequences:
                f.write(f">{sequence_id}\n{sequence}\n")

        autobuild_folder = os.path.join(output_root, "autobuild")
        os.makedirs(autobuild_folder, exist_ok=True)
        os.chdir(autobuild_folder)

        phenix_autobuild_cmd = [
            "phenix.autobuild", 
            f"data={data_path}", 
            autobuild_input_model, 
            master_fasta_filename, 
            f"input_map_file={successful_phaser_map}", 
            "rebuild_in_place=False", 
            "include_input_model=True", 
            f"n_cycle_rebuild_max=5", 
            f"crystal_info.solvent_fraction={solvent_content}", 
            "use_hl_if_present=False",
            f"nproc={args.nproc}"
        ]

        formatted_cmd = ' '.join(phenix_autobuild_cmd)
        logging.info("Formatted phenix_autobuild_cmd for shell execution:")
        logging.info(formatted_cmd)

        # Verify that none of the elements in the command are None
        if any(elem is None for elem in phenix_autobuild_cmd):
            raise ValueError("One or more elements in phenix_autobuild_cmd are None.")

        phenix_autobuild_process = subprocess.Popen(phenix_autobuild_cmd)
        autobuild_log_path = os.path.join(autobuild_folder, "AutoBuild_run_1_/AutoBuild_run_1_1.log")
        # Start the monitoring in a separate thread, passing autobuild_log_path and autobuild_process
        monitor_thread = threading.Thread(target=job_monitor.monitor_and_resolve_hangs, args=(autobuild_log_path, phenix_autobuild_process))
        monitor_thread.start()

        logging.info(f"Monitoring thread started for {autobuild_log_path}.")

        phenix_autobuild_process.wait()
        os.chdir(output_root)

        logging.info("Autobuild process finished.")
    else:
        logging.info("The resolution is worse than 3.5, will use phenix.refine to refine the model.")
        logging.info(f"refinement input model: {autobuild_input_model}")

        refinement_folder = os.path.join(output_root, "refine")
        os.makedirs(refinement_folder, exist_ok=True)
        os.chdir(refinement_folder)
        phenix_refine_cmd = [
            "phenix.refine", 
            autobuild_input_model, 
            data_path, 
            f"strategy=rigid_body+individual_sites+individual_adp", 
            f"main.number_of_macro_cycles=3", 
            f"nproc={args.nproc}"
        ]

        formatted_cmd = ' '.join(phenix_refine_cmd)
        logging.info("Formatted phenix_refine_cmd for shell execution:")
        logging.info(formatted_cmd)

        # Verify that none of the elements in the command are None
        if any(elem is None for elem in phenix_refine_cmd):
            raise ValueError("One or more elements in phenix_refine_cmd are None.")
        
        phenix_refine_process = subprocess.Popen(phenix_refine_cmd)
        phenix_refine_process.wait()

        os.chdir(output_root)

        logging.info("Refinement process finished.")

if __name__ == "__main__":
    main()