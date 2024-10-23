import os
import re
import glob
import logging
import subprocess
import numpy as np
import requests
import json
from Bio.PDB import PDBParser, PDBIO, Chain
from itertools import groupby
from mmtbx.pdbtools import modify, master_params
from pathlib import Path
from SequenceManager import SequenceManager

seq_manager = SequenceManager()

class PDBManager:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        pass
    # Add methods here
    def download_alphafold_model(self, uniprot_id, output_path):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url)
        json_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json"
        json_response = requests.get(json_url)

        if response.status_code == 200: # https://alphafold.ebi.ac.uk/files/AF-P13698-F1-predicted_aligned_error_v4.json
            with open(f"{output_path}/AF-{uniprot_id}-F1-model_v4.pdb", 'wb') as f:
                f.write(response.content)  
            with open(f"{output_path}/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json", 'wb') as f:
                f.write(json_response.content)
                # return the path of the downloaded model and the pae json file
            return f"{output_path}/AF-{uniprot_id}-F1-model_v4.pdb", f"{output_path}/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json"
        elif response.status_code == 404:
            print(f"No AlphaFold model found for UniProt ID {uniprot_id}.")
            return None, None
        else:
            raise ValueError(f"Failed to download AlphaFold model for UniProt ID {uniprot_id}. Status code: {response.status_code}")

    def get_sequence_from_pdb(self, pdb_path):
        """
        Extracts the protein sequence from a PDB file.

        Parameters:
        pdb_path (str): Path to the PDB file.

        Returns:
        str: The extracted protein sequence.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", pdb_path)

        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ' and residue.get_resname() in self.standard_residues:
                        sequence += self.three_to_one(residue.get_resname())

        return sequence

    @staticmethod
    def three_to_one(residue_name):
        """
        Converts a three-letter residue name to a one-letter code.

        Parameters:
        residue_name (str): The three-letter residue name.

        Returns:
        str: The corresponding one-letter code.
        """
        conversion_dict = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
            "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
            "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
            "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
            "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
        }
        return conversion_dict.get(residue_name, "?")

    # Class variable for standard amino acid residues
    standard_residues = set(["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"])

    def truncate_model(self, pdb_path, start, end): # possibly not getting used in main.py, check later
        """
        Truncates the PDB model based on the aligned sequence positions.

        Parameters:
        pdb_path (str): Path to the PDB file of the AlphaFold model.
        start (int): Start index for truncation.
        end (int): End index for truncation.

        Returns:
        str: Path to the truncated model.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", pdb_path)

        for model in structure:
            for chain in model:
                # Remove residues that are not in the aligned range
                residues_to_remove = [residue for residue in chain if residue.id[1] < start or residue.id[1] > end]
                for residue in residues_to_remove:
                    chain.detach_child(residue.id)

        io = PDBIO()
        truncated_path = pdb_path.replace(".pdb", "_truncated.pdb")
        io.set_structure(structure)
        io.save(truncated_path)

        return truncated_path

    def renumber_colabfold_model(self, pdb_path, uniprot_start):
        """
        Renumber the residues of the ColabFold model to match the UniProt numbering.

        Parameters:
        pdb_path (str): Path to the ColabFold PDB file.
        uniprot_start (int): The start position in the UniProt sequence that corresponds to the first residue of the ColabFold model.

        Returns:
        str: Path to the renumbered ColabFold model.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", pdb_path)
        io = PDBIO()

        # First pass: Assign temporary unique IDs to avoid conflicts
        for model in structure:
            for chain in model:
                temp_id = 10000  # Starting from a high number to avoid conflicts
                for residue in chain:
                    residue.id = (' ', temp_id, ' ')
                    temp_id += 1

        # Second pass: Assign the final IDs based on UniProt numbering
        for model in structure:
            for chain in model:
                for residue, new_id in zip(chain, range(uniprot_start, uniprot_start + len(chain))):
                    residue.id = (' ', new_id, ' ')

        renumbered_path = pdb_path.replace(".pdb", "_renumbered.pdb")
        io.set_structure(structure)
        io.save(renumbered_path)

        return renumbered_path

    def process_pdb_file(self, input_pdb, b_factor_cutoff, output_pdb_path):
        # Remove lines not starting with 'ATOM' from the input PDB file    
        with open(input_pdb, 'r') as f_in, open(os.path.splitext(input_pdb)[0] + '_filtered.pdb', 'w') as f_out:
            for line in f_in:
                if line.startswith('ATOM'):
                    f_out.write(line)

        intermediate_pdb = os.path.splitext(input_pdb)[0] + '_filtered.pdb'

        # Initialize a PDBParser object
        parser = PDBParser(QUIET=True)

        cmd = f"phenix.pdbtools {intermediate_pdb} remove='element H or bfactor < {b_factor_cutoff}' output.filename={output_pdb_path}"
        # subprocess.run(cmd, shell=True)
        # Open a null device for output redirection
        with open(os.devnull, 'w') as devnull:
            # Run the command with output and error streams redirected
            subprocess.run(cmd, shell=True, stdout=devnull, stderr=devnull)

        
        # Check if output_pdb_path is created and not empty
        if not os.path.exists(output_pdb_path) or os.stat(output_pdb_path).st_size == 0:
            logging.warning(f"After processing, no atoms above b-factor {b_factor_cutoff} were found in {input_pdb}. Output file {output_pdb_path} is empty.")
            return

        # Parse the structure
        parser = PDBParser(QUIET=True)
        try:
            trimmed = parser.get_structure("protein", output_pdb_path)
        except Exception as e:
            logging.error(f"Error parsing the PDB structure from {output_pdb_path}: {e}")
            return

        # Check for models in the structure
        models = list(trimmed.get_models())
        if not models:
            logging.warning(f"No models found in the trimmed structure {output_pdb_path}.")
            return
        
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

        return output_pdb_path

    def get_prioritized_domains(self, uniprot_id):
        """
        use the interpro API to get the prioritized domains for a given uniprot id
        """
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

                # Check if entry_protein_locations is not None
                if protein["entry_protein_locations"] is not None:
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
        # print(f"Prioritized domains for {uniprot_id}: {filtered_domains}")
        return filtered_domains

    def get_domain_definitions_from_pae(self, pae_json_path, offset=None, primary_pae_cutoff=20, secondary_pae_cutoff=25, gap_threshold=5):
        """
        Get domain definitions from a PAE JSON file in the same structure as the get_domains_from_pdb function.
        
        :param pae_json_path: Path to the PAE JSON file.
        :param pae_cutoff: PAE cutoff value for defining domain boundaries.
        :param gap_threshold: Maximum allowed gap between domains to be merged.
        :return: A list of dictionaries with domain details.
        """
        
        # Load the PAE JSON data
        with open(pae_json_path, 'r') as file:
            pae_data = json.load(file)

        # Determine the structure of the JSON data and extract the PAE matrix
        if isinstance(pae_data, list):
            # New format: list of dictionaries
            pae_matrix = np.array(pae_data[0]['predicted_aligned_error'])
        elif isinstance(pae_data, dict):
            # Old format: dictionary
            pae_matrix = np.array(pae_data['predicted_aligned_error'])
        else:
            raise ValueError("Unexpected PAE JSON file format")

        # Calculate the average PAE for each residue
        avg_pae_per_residue = pae_matrix.mean(axis=1)

        # Define helper functions for identifying and merging domain boundaries
        def identify_domain_boundaries(pae_values, cutoff):
            boundaries = []
            in_domain = pae_values[0] < cutoff
            start = 0 if in_domain else None

            for i, pae in enumerate(pae_values):
                if pae < cutoff and not in_domain:
                    start = i
                    in_domain = True
                elif pae >= cutoff and in_domain:
                    end = i
                    if start is not None:
                        boundaries.append((start, end))
                    in_domain = False
            if in_domain and start is not None:
                boundaries.append((start, len(pae_values)))
            return boundaries

        def merge_close_boundaries(boundaries, gap_threshold):
            merged_boundaries = []
            current_start, current_end = boundaries[0]

            for i in range(1, len(boundaries)):
                next_start, next_end = boundaries[i]
                if next_start - current_end <= gap_threshold:
                    current_end = next_end
                else:
                    merged_boundaries.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged_boundaries.append((current_start, current_end))
            return merged_boundaries

        def update_pae_domains(domains, offset):
            updated_domains = []
            for domain in domains:
                updated_domains.append({
                    "accession": domain["accession"],
                    "name": domain["name"],
                    "start": domain["start"] + offset,
                    "end": domain["end"] + offset
                })
            return updated_domains
        
        # Apply the functions to identify and merge domain boundaries
        # Try identifying domain boundaries with the primary cutoff
        domain_boundaries = identify_domain_boundaries(avg_pae_per_residue, cutoff=primary_pae_cutoff)
        
        # If no domains are found, try with the secondary cutoff
        if not domain_boundaries:
            domain_boundaries = identify_domain_boundaries(avg_pae_per_residue, cutoff=secondary_pae_cutoff)

        merged_domain_boundaries = merge_close_boundaries(domain_boundaries, gap_threshold=gap_threshold)

        # Format the merged domain boundaries in the same structure as get_domains_from_pdb
        domains = []
        for domain_id, (start, end) in enumerate(merged_domain_boundaries, start=1):
            domains.append({
                "accession": f"PAE_DOMAIN_{domain_id}",
                "name": f"Domain {domain_id}",
                "start": start + 1,  # Adjust for 1-based indexing used in PDB files
                "end": end
            })

        # If an offset is provided, update the domain boundaries
        if offset is not None:
            domains = update_pae_domains(domains, offset)

        return domains

    def extract_domain_from_pdb(self, pdb_file, start, end, output_file):
        parser = PDBParser(QUIET=True)
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

    def prepare_domain_ensembles(self, input_pdb, domains, domain_output_dir, sequence_length):
        if domains is not None:
            # first prepare the domain isolated ensembles 
            # domains = get_prioritized_domains(uniprot_id)
            Path(domain_output_dir).mkdir(parents=True, exist_ok=True)
            # Calculate the threshold length (20% of the input sequence length)
            threshold_length = 0.2 * sequence_length
            # if threshold_length > 100, then set it to 100
            if threshold_length > 80:
                threshold_length = 80
            # logging.info(f"Threshold length for a domain to be used: {int(round(threshold_length))}")
            sorted_domains = sorted(domains, key=lambda x: x['end'] - x['start'], reverse=True)
            # Adjust domain boundaries
            adjusted_domains = seq_manager.adjust_domain_boundaries(sorted_domains, sequence_length)
            # logging.info(f"List of adjusted domains: {adjusted_domains}")
            
            ensemble_files = []
            input_seq_start, input_seq_end = self.get_pdb_sequence_range(input_pdb)
            # logging.info(f"Input sequence range: {input_seq_start} - {input_seq_end}")
            # Extract the domains from the PDB file and save them as ensembles
            for i, domain in enumerate(adjusted_domains):
                domain_start, domain_end = max(domain['start'], input_seq_start), min(domain['end'], input_seq_end)
                # domain_length = domain['end'] - domain['start'] + 1
                domain_length = domain_end - domain_start + 1

                # Check if the domain length is greater than or equal to 20% of the input sequence length
                if domain_length >= threshold_length and domain_start <= domain_end:
                    # logging.info(f"Defined Domain {i+1}: {domain['start']} - {domain['end']}")
                    # logging.info(f"Domain {i+1} boundary: {domain_start} - {domain_end}")
                    logging.info(f"Domain {i+1} will be used for MR with length: {domain_length}")
                    output_file = os.path.join(domain_output_dir, f"ensemble_{i+1}.pdb")
                    ensemble_files.append(output_file)
                    self.extract_domain_from_pdb(input_pdb, domain_start, domain_end, output_file)
                else:
                    # If the domain length is less than 20% of the input sequence length, stop processing further domains
                    # logging.info(f"Defined Domain {i+1}: {domain['start']} - {domain['end']}")
                    # logging.info(f"Domain {i+1} boundary: {domain_start} - {domain_end}")
                    # logging.info(f"Domain length: {domain_length} less than {int(round(threshold_length))}. Skipping this domain.")
                    continue
        else:
            ensemble_files = [input_pdb]
        
        return ensemble_files

    def get_pdb_sequence_range(self, pdb_path):
        # Extracts the first and last residue numbers from the PDB file
        with open(pdb_path, 'r') as file:
            first_residue, last_residue = None, None
            for line in file:
                if line.startswith("ATOM"):
                    residue_number = int(line[22:26].strip())  # Extracting residue number from the PDB ATOM line
                    if first_residue is None:
                        first_residue = residue_number
                    last_residue = residue_number
            return first_residue, last_residue


    def get_next_pdb_entry(self, file_path, used_pdbs):
        with open(file_path, "r") as file:
            lines = file.readlines()
            # skip the first line
            lines = lines[1:]
            for line in lines:
                pdb = line.split(",")[0]
                if pdb not in used_pdbs:
                    return pdb
        return None  

    def get_sequence_length_from_pdb(self, pdb_path):
        sequence_length = 0
        residues = set()
        
        try:
            with open(pdb_path, 'r') as pdb_file:
                for line in pdb_file:
                    if line.startswith("SEQRES"):
                        parts = line.split()
                        sequence_length += len(parts) - 4
                    elif line.startswith("ATOM"):
                        residue_id = line[17:26]  # Extract residue name, chain identifier, and residue sequence number
                        residues.add(residue_id)
            
            if sequence_length == 0:
                sequence_length = len(residues)
        except FileNotFoundError:
            logging.error(f"PDB file not found: {pdb_path}")
        except Exception as e:
            logging.error(f"An error occurred while reading the PDB file: {e}")
        
        return sequence_length

    """
    This part is for multiple sequences input cases
    """
    def parse_phaser_log(self, log_file_path: str) -> list:
        """Parse the PHASER log file to find the first solution block, extract relevant solutions,
        calculate CC per chain, and adjust the keep flags based on CC values."""

        solutions = []
        first_solution_block_found = False
        tncs_present = False
        have_pre_placed_chains = False
        in_solu_set = False

        pdb_file_path = os.path.join(os.path.dirname(log_file_path), "PHASER.1.pdb")
        mtz_file_path = os.path.join(os.path.dirname(log_file_path), "PHASER.1.mtz")

        # Store LLG and TFZ pairs
        llg_tfz_pairs = []
        single_solution = False
        solu_set_lines = []

        with open(log_file_path, 'r') as file:
            for line in file:
                if '** SINGLE solution' in line or 'Solution annotation (history):' in line or 'Solution #1 annotation (history):' in line or 'Partial Solution #1 annotation (history):' in line:
                    first_solution_block_found = True
                    if "** SINGLE solution" in line:
                        single_solution = True
                elif first_solution_block_found:
                    if 'SOLU SET' in line:
                        in_solu_set = True
                    elif "SOLU SPAC" in line:
                        in_solu_set = False
                    if in_solu_set:
                        solu_set_lines.append(line.strip())
                    if 'Solution #' in line:
                        break

        solu_set_combined = ' '.join(solu_set_lines)

        # Custom parser for the SOLU SET line
        def parse_solu_set_line(solu_set_line):
            # Preprocess the line to normalize it
            line = solu_set_line.replace('(&', '&')
            line = line.replace(')', '')
            line = line.replace('(', '')
            line = line.replace('&', ' & ')
            line = re.sub(r'\s+', ' ', line)  # Replace multiple spaces with a single space
            # print(f"Preprocessed line: {line}")

            tokens = line.strip().split(' ')
            # print(f"Tokens: {tokens}")  # Debug print to check tokens

            LLG_TFZ_pairs = []
            current_LLGs = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token.startswith('LLG=') or token.startswith('LLG+='):
                    # Extract LLG value(s)
                    llg_values = []
                    llg_str = token.split('=')[1]
                    llg_parts = [llg_str]
                    idx += 1
                    while idx < len(tokens) and (tokens[idx].isdigit() or tokens[idx] == '&'):
                        if tokens[idx] != '&':
                            llg_parts.append(tokens[idx])
                        idx += 1
                    for llg_part in llg_parts:
                        if llg_part.isdigit():
                            llg_values.append(int(llg_part))
                    current_LLGs = llg_values
                    # print(f"Current LLGs: {current_LLGs}")  # Debug print to check current LLGs
                elif token.startswith('TFZ=='):
                    # Extract TFZ value(s)
                    tfz_values = []
                    tfz_str = token.split('==')[1]
                    tfz_parts = [tfz_str]
                    idx += 1
                    while idx < len(tokens) and (re.match(r'^\d+\.?\d*$', tokens[idx]) or tokens[idx] == '&'):
                        if tokens[idx] != '&':
                            tfz_parts.append(tokens[idx])
                        idx += 1
                    for tfz_part in tfz_parts:
                        if re.match(r'^\d+\.?\d*$', tfz_part):
                            tfz_values.append(float(tfz_part))
                    # print(f"TFZ values: {tfz_values}")  # Debug print to check TFZ values
                    for tfz in tfz_values:
                        for llg in current_LLGs:
                            LLG_TFZ_pairs.append((llg, tfz))
                            # print(f"Added pair: {(llg, tfz)}")  # Debug print to check added pairs
                else:
                    idx += 1
            return LLG_TFZ_pairs
        # Use the custom parser to extract LLG and TFZ pairs
        llg_tfz_pairs = parse_solu_set_line(solu_set_combined)

        if '+TNCS' in solu_set_combined:
            tncs_present = True
        # logging.info(f"llg_tfz_pairs: {llg_tfz_pairs}")

        solutions = []
        first_solution_block_found = False
        have_pre_placed_chains = False
        in_solu_set = False
        with open(log_file_path, 'r') as file:
            for line in file:
                if '** SINGLE solution' in line or 'Solution annotation (history):' in line or 'Solution #1 annotation (history):' in line or 'Partial Solution #1 annotation (history):' in line:
                    first_solution_block_found = True
                    if "** SINGLE solution" in line:
                        single_solution = True
                elif first_solution_block_found:
                    if 'SOLU SET' in line:
                        in_solu_set = True
                    elif "SOLU SPAC" in line:
                        in_solu_set = False
                    if not in_solu_set and 'SOLU 6DIM ENSE' in line:
                        if 'EULER    0.0    0.0    0.0' in line:
                            have_pre_placed_chains = True
                        else:
                            ensemble_id_match = re.search(r'ENSE\s+(\S+)', line)
                            if ensemble_id_match:
                                ensemble_id = ensemble_id_match.group(1)
                                keep = False
                                tfz_match = re.search(r'#TFZ==(\d+\.\d+)', line)
                                if tncs_present:
                                    keep = True
                                else:
                                    if tfz_match:
                                        tfz_score = float(tfz_match.group(1))
                                        if single_solution:
                                            llg_threshold = 40.0
                                        else:
                                            llg_threshold = 40.0
                                        # Check if any pair meets the condition int(llg) >= int(llg_threshold) and float(tfz) >= 8.0
                                        valid_pair_exists = any(int(llg) >= int(llg_threshold) and float(tfz) >= 8.0 for llg, tfz in llg_tfz_pairs)
                                        for llg, tfz in llg_tfz_pairs:
                                            if float(tfz) == tfz_score:
                                                if (int(llg) >= int(llg_threshold) and float(tfz) >= 8.0) or (single_solution and float(tfz) >= 6.0 and not valid_pair_exists):
                                                    keep = True
                                solutions.append((ensemble_id, keep))
                    if 'Solution #' in line:
                        break  # Stop processing after the first solution block

        # Adjust for any pre-placed chains by comparing the number of chains in PHASER.1.pdb
        if have_pre_placed_chains:
            solutions = self.adjust_for_pre_placed_chains(pdb_file_path, solutions)
        print(f"PHASER solutions before CC filtering: {solutions}")

        # Now, if PHASER.1.pdb and PHASER.1.mtz exist, calculate CC per chain and adjust keep flags
        if os.path.exists(pdb_file_path) and os.path.exists(mtz_file_path):
            # Calculate CC per chain
            cc_per_chain, _ = self.calculate_cc_per_chain(pdb_file_path, mtz_file_path)

            # Build a mapping from chain_id to cc value
            cc_per_chain_dict = {chain_cc['chain_id']: chain_cc['cc'] for chain_cc in cc_per_chain}
            print(f"cc_per_chain_dict: {cc_per_chain_dict}")
            # Get chain IDs from the PDB file
            chain_ids = self.get_chain_ids_from_pdb(pdb_file_path)

            # Adjust solutions based on CC values
            for i, (ensemble_id, keep) in enumerate(solutions):
                print(f"Checking chain {i} with ensemble_id {ensemble_id}")
                if i < len(chain_ids):
                    chain_id = chain_ids[i]
                    cc = cc_per_chain_dict.get(chain_id)
                    if cc is not None and cc < 0.45:
                        solutions[i] = (ensemble_id, False)
                else:
                    # If chain IDs are fewer than solutions, keep as is or handle accordingly
                    pass

        logging.info(f"PHASER solutions after CC filtering: {solutions}")
        return solutions
    
    def adjust_for_pre_placed_chains(self, pdb_file_path: str, solutions: list) -> list:
        """Adjust the solutions list based on the number of chains in PHASER.1.pdb."""
        with open(pdb_file_path, 'r') as pdb_file:
            chains_in_pdb = set()
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chains_in_pdb.add(line[21])

        pre_placed_chains = len(chains_in_pdb) - len(solutions)
        if pre_placed_chains < 0:
            raise ValueError("More solutions found in the log file than chains in the PDB file.")

        # Prepend (None, False) for pre-placed chains
        adjusted_solutions = [(None, False)] * pre_placed_chains + solutions
        return adjusted_solutions


    def process_pdb_file_for_phaser(self, pdb_file_path, keep_chains, output_file_path, partial_pdb_path=None):
        """Process the PDB file to keep only the relevant chains, reassign chain identifiers, and remove residues with any atom occupancy less than 1."""
        print(f"kept chains: {keep_chains}")
        with open(pdb_file_path, 'r') as pdb_file, open(output_file_path, 'w') as output_file:
            chain_map = {}
            next_chain_id = ord('A')  # Start with 'A'

            # If a partial PDB path is provided, read chain identifiers from it
            if partial_pdb_path is not None:
                with open(partial_pdb_path, 'r') as partial_pdb_file:
                    existing_chains = set()
                    for line in partial_pdb_file:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            existing_chains.add(line[21])
                    if existing_chains:
                        last_chain_id = sorted(existing_chains)[-1]
                        if last_chain_id.upper() < 'Z':
                            next_chain_id = ord(last_chain_id.upper()) + 1
                        else:
                            raise ValueError("Cannot assign new chain identifiers beyond 'Z'.")

            # Extract only the boolean values from keep_chains tuples
            keep_flags = [keep for _, keep in keep_chains]

            # Write the existing partial PDB content first, if provided
            if partial_pdb_path is not None:
                output_file.write(open(partial_pdb_path).read())

            # Build a set of residues to delete based on cc_per_residue
            mtz_file_path = os.path.join(os.path.dirname(pdb_file_path), "PHASER.1.mtz")
            _, cc_per_residue = self.calculate_cc_per_chain(pdb_file_path, mtz_file_path)
            residues_to_delete = set()
            if cc_per_residue is not None:
                for res in cc_per_residue:
                    chain_id = res['chain_id']
                    resseq = res['resseq'].strip()
                    residues_to_delete.add((chain_id, resseq))

            residue_lines = []
            current_residue_id = None

            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain = line[21]
                    chain_index = ord(chain) - ord('A')
                    resseq = line[22:27].strip()  # Include insertion codes
                    residue_id = (resseq, chain)  # (resseq, chain_id)

                    if residue_id != current_residue_id:
                        # Before processing new residue, write out the previous one if needed
                        if residue_lines and all(float(l[54:60].strip()) >= 1.0 for l in residue_lines):
                            previous_resseq = current_residue_id[0]
                            previous_chain = current_residue_id[1]
                            res_tuple = (previous_chain, previous_resseq)
                            if res_tuple not in residues_to_delete:
                                for l in residue_lines:
                                    output_file.write(l)
                        residue_lines = []
                        current_residue_id = residue_id

                    if chain_index < len(keep_flags) and keep_flags[chain_index]:
                        if chain not in chain_map:
                            chain_map[chain] = chr(next_chain_id)
                            next_chain_id += 1
                        new_chain = chain_map[chain]
                        # Update the chain ID in the line to new_chain
                        new_line = line[:21] + new_chain + line[22:]
                        residue_lines.append(new_line)
                else:
                    # Non-ATOM/HETATM lines
                    if residue_lines and all(float(l[54:60].strip()) >= 1.0 for l in residue_lines):
                        previous_resseq = current_residue_id[0]
                        previous_chain = current_residue_id[1]
                        res_tuple = (previous_chain, previous_resseq)
                        if res_tuple not in residues_to_delete:
                            for l in residue_lines:
                                output_file.write(l)
                    residue_lines = []
                    current_residue_id = None
                    output_file.write(line)

            # Write any remaining residue
            if residue_lines and all(float(l[54:60].strip()) >= 1.0 for l in residue_lines):
                previous_resseq = current_residue_id[0]
                previous_chain = current_residue_id[1]
                res_tuple = (previous_chain, previous_resseq)
                if res_tuple not in residues_to_delete:
                    for l in residue_lines:
                        output_file.write(l)

    def get_chain_ids_from_pdb(self, pdb_file_path: str) -> list:
        """Extract chain IDs from a PDB file in the order they appear."""
        chain_ids = []
        with open(pdb_file_path, 'r') as pdb_file:
            for line in pdb_file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    chain_id = line[21]
                    if chain_id not in chain_ids:
                        chain_ids.append(chain_id)
        return chain_ids
    
    def calculate_cc_per_chain(self, pdb_file_path, mtz_file_path, d_min=None):
        """Calculate map-model CC for each chain in the PDB file."""
        from iotbx import reflection_file_reader, pdb
        from cctbx import miller
        from mmtbx.maps.map_model_cc import map_model_cc, master_params

        def identify_map_coefficients(miller_arrays):
            amplitude_labels = ['FWT', '2FOFCWT', 'FOBS', 'FP', 'F', 'F_ML', 'FOBS(+)', 'F-OBS', 'F-MODEL', '2FOFCWT_NO_FILL', 'FOFCWT']
            phase_labels = ['PHWT', 'PH2FOFCWT', 'PHIB', 'PHI', 'PHIM', 'PHASE', 'PHIF-MODEL', 'PHFOFCWT', 'PH2FOFCWT_NO_FILL']

            label_to_array = {}
            for ma in miller_arrays:
                labels = ma.info().labels
                for label in labels:
                    label_upper = label.upper()
                    label_to_array[label_upper] = ma

            amplitude_arrays = []
            phase_arrays = []
            for label in amplitude_labels:
                if label in label_to_array:
                    amplitude_arrays.append((label, label_to_array[label]))
            for label in phase_labels:
                if label in label_to_array:
                    phase_arrays.append((label, label_to_array[label]))

            for amp_label, amp_array in amplitude_arrays:
                for phase_label, phase_array in phase_arrays:
                    if amp_array.indices().all_eq(phase_array.indices()):
                        return amp_array, phase_array, amp_label, phase_label
            return None, None, None, None

        # Read PDB and MTZ files
        pdb_inp = pdb.input(file_name=pdb_file_path)
        pdb_hierarchy = pdb_inp.construct_hierarchy()
        xray_structure = pdb_inp.xray_structure_simple()
        cs = xray_structure.crystal_symmetry()

        reflection_file = reflection_file_reader.any_reflection_file(mtz_file_path)
        miller_arrays = reflection_file.as_miller_arrays(crystal_symmetry=cs)

        # Identify amplitude and phase arrays
        f_obs, phases, f_label, phi_label = identify_map_coefficients(miller_arrays)

        if f_obs is None or phases is None:
            print("Available labels in the MTZ file:")
            for ma in miller_arrays:
                print(ma.info().labels)
            raise ValueError("Could not automatically identify amplitude and phase labels in the MTZ file.")

        # Ensure f_obs is an amplitude array
        if not f_obs.is_xray_amplitude_array():
            if f_obs.is_complex_array():
                f_obs = f_obs.amplitudes()
            elif f_obs.is_xray_intensity_array():
                f_obs = f_obs.f_sq_as_f()
            elif f_obs.is_real_array():
                pass
            else:
                raise ValueError(f"Array {f_label} is not an amplitude array and cannot be converted.")

        # Ensure phases are real
        if not phases.is_real_array():
            if phases.is_complex_array():
                phases = phases.phases()
            elif phases.is_real_array():
                phases = miller.array(miller_set=phases.set(), data=phases.data(), sigmas=None)
                phases.set_observation_type_xray_phase_deg()
            else:
                raise ValueError(f"Array {phi_label} is not a phase array and cannot be converted.")

        # Generate map coefficients from observed amplitudes and phases
        map_coeffs = f_obs.phase_transfer(phase_source=phases)

        # If d_min is not provided, use the resolution from the data
        if d_min is None:
            d_min = f_obs.d_min()

        if d_min is not None:
            map_coeffs = map_coeffs.resolution_filter(d_min=d_min)

        # Generate real-space map from map coefficients
        fft_map = map_coeffs.fft_map(resolution_factor=0.25)
        fft_map.apply_sigma_scaling()
        map_data = fft_map.real_map_unpadded()

        # Compute Map-Model CC using mmtbx.maps.map_model_cc
        params = master_params().extract().map_model_cc
        if d_min is not None:
            params.resolution = d_min
        params.compute.cc_mask = False
        params.compute.cc_volume = False
        params.compute.cc_peaks = False
        params.compute.cc_box = False
        params.compute.cc_per_chain = True
        params.compute.cc_per_residue = True
        cc_calculator = map_model_cc(
            map_data=map_data,
            pdb_hierarchy=pdb_hierarchy,
            crystal_symmetry=cs,
            params=params
        )
        cc_calculator.validate()
        cc_calculator.run()
        results = cc_calculator.get_results()

        # Store results in a list
        cc_per_chain = []
        for chain in results.cc_per_chain:
            cc_per_chain.append({
                "chain_id": chain.chain_id,
                "cc": chain.cc,
                "n_atoms": chain.n_atoms,
                "b_iso_mean": chain.b_iso_mean,
                "occ_mean": chain.occ_mean
            })

        cc_per_residue = []
        for residue in results.cc_per_residue:
            if residue.cc < 0.6:
                
                cc_per_residue.append({
                    "chain_id": residue.chain_id,
                    "resseq": residue.resseq,            
                    "cc": residue.cc,
                })
        return cc_per_chain, cc_per_residue