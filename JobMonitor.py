import signal
import subprocess
import os
import re
import time
import logging
import psutil
from utilities import get_cpu_usage


class JobMonitor:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)

    def mark_colabfold_finished(self, output_dir, structure_name):
        done_file = os.path.join(output_dir, f"{structure_name}.done.txt")
        return os.path.exists(done_file)

    def monitor_and_resolve_hangs(self, autobuild_log_path, autobuild_process):
        self._monitor(autobuild_log_path, autobuild_process, self.check_and_resolve_hangs)

    def monitor_and_resolve_memory_leaks(self, autobuild_log_path, autobuild_process):
        self._monitor(autobuild_log_path, autobuild_process, self.check_and_resolve_memory_leaks)

    def _monitor(self, autobuild_log_path, autobuild_process, check_method):
        while autobuild_process.poll() is None:
            check_method(autobuild_log_path)
            time.sleep(600 if check_method == self.check_and_resolve_hangs else 60)

    def check_and_resolve_hangs(self, autobuild_log_path):
        job_paths = self._extract_job_paths(autobuild_log_path)
        if job_paths:
            self._resolve_job_block(job_paths, resolve_hangs=True)

    def check_and_resolve_memory_leaks(self, autobuild_log_path):
        info_file_paths, missing_info_file_paths, job_paths, missing_job_paths = self._extract_info_and_job_paths(autobuild_log_path)
        if info_file_paths:
            self._resolve_memory_leaks(info_file_paths)
        if missing_info_file_paths: # deal with missing info files causing memory leaks
            self._resolve_memory_leaks(missing_info_file_paths)
        if missing_job_paths:
            self.logger.warning(f"Missing job paths: {missing_job_paths}")
            for missing_job_path in missing_job_paths:
                # check if the missing job path corresponds to a missing info file
                self.logger.warning(f"Created missing job path: {missing_job_path}")
                stopwizard_file = os.path.join(missing_job_path, 'STOPWIZARD')
                try:
                    with open(stopwizard_file, 'w') as file:
                        file.write('STOPWIZARD')
                except Exception as e:
                    self.logger.error(f"Error creating STOPWIZARD file in missing job path {missing_job_path}: {e}")
                go_on_file = os.path.dirname(autobuild_log_path) + '/GO_ON'
                list_of_jobs_running_file = os.path.dirname(autobuild_log_path) + '/LIST_OF_JOBS_RUNNING'
                try:
                    final_count_pattern = r"FINAL COUNT:\s+(\d+)"
                    with open(list_of_jobs_running_file, 'r') as file:
                        final_count = None
                        for line in file:
                            match = re.search(final_count_pattern, line)
                            if match:
                                final_count = int(match.group(1))
                                break
                        if final_count is not None:
                            self.logger.info(f"Final count: {final_count}")
                            with open(go_on_file, 'w') as file:
                                file.write(final_count)
                        else:
                            self.logger.warning("Final count not found in LIST_OF_JOBS_RUNNING file.")
                except Exception as e:
                    self.logger.error(f"Error creating GO_ON file in missing job path {missing_job_path}: {e}")


    def _extract_job_paths(self, autobuild_log_path):
        try:
            with open(autobuild_log_path, 'r') as log_file:
                log_contents = log_file.read()
            return self._parse_log_for_job_paths(log_contents)
        except Exception as e:
            self.logger.error(f"Error extracting job paths: {e}")
            return []

    def _extract_info_and_job_paths(self, autobuild_log_path):
        try:
            with open(autobuild_log_path, 'r') as log_file:
                log_contents = log_file.read()
            return self._parse_log_for_info_and_job_paths(log_contents)
        except Exception as e:
            self.logger.error(f"Error extracting info and job paths: {e}")
            return [], []

    def _parse_log_for_job_paths(self, log_contents):
        job_logs = re.findall(r"Log will be: (\/[\w\/.-]+\/RUN_FILE_(\d+)\.log)", log_contents)
        job_paths = []
        for log, job_number in job_logs:
            base_path = os.path.dirname(log)
            info_file_path = os.path.join(base_path, f"INFO_FILE_{job_number}")
            with open(info_file_path, 'r') as info_file:
                job_path = info_file.read().strip()
            if job_path not in job_paths:
                job_paths.append(job_path)
        return job_paths

    def _parse_log_for_info_and_job_paths(self, log_contents):
        job_logs = re.findall(r"Log will be: (\/[\w\/.-]+\/RUN_FILE_(\d+)\.log)", log_contents)
        info_file_paths = []
        missing_info_file_paths = []
        job_paths = []
        missing_job_paths = []
        for log, job_number in job_logs:
            base_path = os.path.dirname(log)
            info_file_path = os.path.join(base_path, f"INFO_FILE_{job_number}")
            if not os.path.exists(info_file_path):
                if info_file_path not in missing_info_file_paths:
                    missing_info_file_paths.append(info_file_path)
            else:
                if info_file_path not in info_file_paths:
                    info_file_paths.append(info_file_path)
            with open(info_file_path, 'r') as info_file:
                job_path = info_file.read().strip()
            if job_path not in job_paths:
                job_paths.append(job_path)
            if not os.path.exists(job_path):
                missing_job_paths.append(job_path)
        return info_file_paths, missing_info_file_paths, job_paths, missing_job_paths

    def _resolve_job_block(self, job_paths, resolve_hangs=False):
        finished_jobs = [job for job in job_paths if os.path.exists(os.path.join(job, 'FINISHED'))]
        if len(job_paths) - len(finished_jobs) == 1:
            hanging_job = set(job_paths) - set(finished_jobs)
            for job in hanging_job:
                self.create_finished_file(job)
                self.logger.warning(f"Created FINISHED file for {'hanging' if resolve_hangs else 'leaking memory'} job: {job}")

    def _resolve_memory_leaks(self, info_file_paths):
        for info_file_path in info_file_paths:
            pid = self.get_cmd_from_info_file(info_file_path)
            if pid:
                memory_usage = self.get_cmd_memory_usage(pid)
                if memory_usage > (psutil.virtual_memory().total * 0.25):
                    with open(info_file_path, 'r') as info_file:
                        job_path = info_file.read().strip()
                    self.create_finished_file(job_path)
                    self.logger.warning(f"Memory leak detected and resolved for job {pid}: {job_path}")
                    try:
                        time.sleep(20)
                        os.kill(int(pid), signal.SIGTERM)
                        self.logger.info(f"Terminated process with PID {pid}")
                    except OSError as e:
                        self.logger.error(f"Error terminating process with PID {pid}: {e}")

    def get_cmd_from_info_file(self, info_file_path):
        """Find the system-wide command that contains the whole value of the info_file_path and get the pid."""
        try:
            cmd = f"ps aux | grep {info_file_path} | grep -v grep"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, _ = process.communicate()
            if output:
                return output.decode('utf-8').split()[1]
        except Exception as e:
            self.logger.error(f"Error retrieving PID from info file: {e}")
        return None

    def get_cmd_memory_usage(self, pid):
        """Get the memory usage of the process with the given PID."""
        try:
            process = psutil.Process(int(pid))
            return process.memory_info().rss
        except (psutil.NoSuchProcess, ValueError) as e:
            self.logger.error(f"Error retrieving memory usage for PID {pid}: {e}")
        return 0

    def create_finished_file(self, job):
        finished_file = os.path.join(job, 'FINISHED')
        try:
            open(finished_file, 'w').close()
        except Exception as e:
            self.logger.error(f"Error creating FINISHED file for job {job}: {e}")