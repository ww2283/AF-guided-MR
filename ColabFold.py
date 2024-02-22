import os
import time
import subprocess
from utilities import wait_for_available_gpu

class ColabFold:

    def __init__(self) -> None:
        pass

    def run_colabfold(self, input_csv, output_dir, num_models=1, num_recycle=5, amber=True, use_gpu_relax=True):
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