# utilities.py
import logging
import os
import csv
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

def setup_custom_logger():
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

    def success(msg, *args, **kwargs):
        logging.log(25, msg, *args, **kwargs)

    def fail(msg, *args, **kwargs):
        logging.log(45, msg, *args, **kwargs)

    logging.success = success
    logging.fail = fail

def save_csv_report(output_file, num_sequences, sequence_length, run_time, resolution, tfz_score, successful_phaser_run, successful_phaser_output_dir, mr_rosetta_success):
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['num_sequences', 'sequence_length', 'run_time', 'resolution', 'tfz_score', 'successful_phaser_run', 'successful_phaser_output_dir', 'mr_rosetta_success']
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
            'mr_rosetta_success': mr_rosetta_success
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
    