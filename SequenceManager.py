import csv
import requests
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Blast import NCBIWWW, NCBIXML
from multiprocessing import Process, Queue
import logging

class SequenceManager:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        pass

    def get_uniprot_id_from_sequence(self, sequence):
        try:
            result_handle = NCBIWWW.qblast("blastp", "swissprot", sequence)
            blast_records = NCBIXML.parse(result_handle)

            for blast_record in blast_records:
                for alignment in blast_record.alignments:
                    uniprot_id = alignment.accession.split(".")[0]
                    return uniprot_id
        except Exception as e:
            self.logger.error(f"Error in get_uniprot_id_from_sequence: {e}")
            return None

    def fetch_uniprot_id(self, sequence, result_queue):
        uniprot_id = self.get_uniprot_id_from_sequence(sequence)
        result_queue.put(uniprot_id)

    def get_uniprot_id_with_timeout(self, sequence, timeout=200):
        result_queue = Queue()
        process = Process(target=self.fetch_uniprot_id, args=(sequence, result_queue))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            self.logger.warning("Timed out while trying to fetch UniProt ID.")
            return None
        else:
            return result_queue.get()

    def get_uniprot_sequence(self, uniprot_id):
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            fasta_text = response.text
            # The first line of a fasta file starts with '>', so we remove it
            # and join the rest which is the actual sequence
            sequence = ''.join(fasta_text.split('\n')[1:])
            return sequence
        else:
            print(f"Failed to get data from UniProtKB for ID {uniprot_id}.")
            return None

            

    def get_absolute_positions(self, input_seq, uniprot_id):
        """
        Aligns the input sequence with the full UniProt sequence and returns the absolute positions of the most significant alignment.

        Parameters:
        input_seq (str): The input sequence (part of the UniProt sequence).
        uniprot_id (str): The UniProt ID for the full sequence.

        Returns:
        tuple: A tuple containing the absolute start and end indices of the input sequence within the UniProt sequence.
        """
        uniprot_seq = self.get_uniprot_sequence(uniprot_id)

        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1  # High match score
        aligner.mismatch_score = -0.1  # Less punitive mismatch score
        aligner.open_gap_score = -0.5  # Penalty for opening a gap
        aligner.extend_gap_score = -0.1  # Penalty for extending a gap

        alignments = aligner.align(input_seq, uniprot_seq)

        # Find the most significant alignment segment
        longest_segment = (0, 0)
        longest_length = 0
        for aligned_input, aligned_uniprot in zip(alignments[0].aligned[0], alignments[0].aligned[1]):
            segment_length = aligned_uniprot[1] - aligned_uniprot[0]
            if segment_length > longest_length:
                longest_length = segment_length
                longest_segment = aligned_uniprot

        start, end = longest_segment

        return start, end  # end is inclusive and adjusted

        
    def align_sequences(self, input_seq, model_seq):
        """
        Aligns the input sequence with the AlphaFold model sequence.

        Parameters:
        input_seq (str): The input sequence (part of the UniProt sequence).
        model_seq (str): The full sequence from the AlphaFold model.

        Returns:
        tuple: A tuple containing the start and end indices of the model sequence that align with the input sequence.
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1  # High match score
        aligner.mismatch_score = -0.1  # Less punitive mismatch score
        aligner.open_gap_score = -0.5  # Penalty for opening a gap
        aligner.extend_gap_score = -0.1  # Penalty for extending a gap

        alignments = aligner.align(input_seq, model_seq)

        # Print the best alignment
        # print(alignments[0])

        # Find the most significant alignment segment
        longest_segment = (0, 0)
        longest_length = 0
        for aligned_input, aligned_model in zip(alignments[0].aligned[0], alignments[0].aligned[1]):
            segment_length = aligned_model[1] - aligned_model[0]
            if segment_length > longest_length:
                longest_length = segment_length
                longest_segment = aligned_model

        start, end = longest_segment

        return start, end

    def adjust_domain_boundaries(self, domains, sequence_length):
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
    

    """
    the following part is for multiple sequences input cases
    """
    def read_sequences_from_csv(self, csv_path):
        sequences = []
        try:
            with open(csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                if header[:2] != ['id', 'sequence']:
                    raise ValueError("CSV header should start with 'id,sequence'")
                for row in reader:
                    if len(row) < 2:
                        raise ValueError("Each row in CSV must have at least 2 elements: 'id' and 'sequence'")
                    structure_id, sequence = row[:2]  # Only read the first two elements
                    sequences.append((structure_id, sequence))
        except Exception as e:
            self.logger.error(f"Error in read_sequences_from_csv: {e}")
            return None

        return sequences
    
    def write_custom_sequence_to_fasta(self, structure_name, sequence, fasta_filename):
        try:
            with open(fasta_filename, "w") as fasta_file:
                fasta_file.write(f">{structure_name}\n")
                fasta_file.write(sequence)
        except Exception as e:
            self.logger.error(f"Error in write_sequence_to_fasta: {e}")
