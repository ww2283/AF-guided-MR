import subprocess
import os
import re
import time
import threading
import logging
from utilities import get_cpu_usage

class JobMonitor:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)

    def mark_colabfold_finished(self, output_dir, structure_name):
        done_file = os.path.join(output_dir, f"{structure_name}.done.txt")
        return os.path.exists(done_file)

    def check_and_resolve(self, autobuild_log_path):
        try:
            with open(autobuild_log_path, 'r') as log_file:
                log_contents = log_file.read()

            job_logs = re.findall(r"Log will be: (\/[\w\/.-]+\/RUN_FILE_(\d+)\.log)", log_contents)
            job_paths = []
            for log, job_number in job_logs:
                base_path = os.path.dirname(log)
                job_path = os.path.join(base_path, f"AutoBuild_run_{job_number}_")
                job_paths.append(job_path)

            if job_paths:
                self.check_and_resolve_job_block(job_paths)
        except Exception as e:
            print(f"Error during monitoring: {e}")

    def check_and_resolve_job_block(self, job_paths):
        finished_jobs = [job for job in job_paths if os.path.exists(os.path.join(job, 'FINISHED'))]
        if len(job_paths) - len(finished_jobs) == 1:
            hanging_job = set(job_paths) - set(finished_jobs)
            for job in hanging_job:
                open(os.path.join(job, 'FINISHED'), 'w').close()
                logging.warning(f"Created FINISHED file for hanging job: {job}")