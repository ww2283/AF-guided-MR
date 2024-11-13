import concurrent.futures
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from queue import Queue, Empty
import logging
import utilities
import os
import glob
import subprocess

@dataclass
class RefinementResult:
    cluster_number: int
    r_work: float 
    r_free: float
    refinement_folder: str
    partial_pdb_path: str
    phaser_output_dir: str
    mode: str
    tfz_score: float
    is_complete: bool = False
    process: Optional[subprocess.Popen] = None 

    @property
    def phaser_output_map(self) -> str:
        """Get the path to the phaser output map file"""
        return glob.glob(os.path.join(self.refinement_folder, '*.mtz'))[0]

class AsyncRefinementManager:
    def __init__(self, r_free_threshold: float):
        self.refinement_futures: Dict[int, concurrent.futures.Future] = {}
        self.refinement_results: Dict[int, RefinementResult] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.r_free_threshold = r_free_threshold
        self.success_queue = Queue()


    def start_refinement(self, cluster_number: int, partial_pdb_path: str, mtz_path: str, 
                        refine_output_root: str, nproc: int, phaser_output_dir: str,
                        mode: str = "AF_cluster_mode", tfz_score: float = 0.0):
        future = self.executor.submit(
            self._run_refinement,
            cluster_number,
            partial_pdb_path,
            mtz_path,
            refine_output_root,
            nproc,
            phaser_output_dir,
            mode,
            tfz_score
        )
        self.refinement_futures[cluster_number] = future

    def _run_refinement(self, cluster_number: int, partial_pdb_path: str, mtz_path: str, 
                        refine_output_root: str, nproc: int, phaser_output_dir: str,
                        mode: str = "AF_cluster_mode", tfz_score: float = 0.0) -> RefinementResult:
        try:
            # Run refinement and get results
            r_work, r_free, refinement_folder, process = utilities.rfactors_from_phenix_refine(
                partial_pdb_path, mtz_path, refine_output_root, nproc=nproc
            )

            # Create refinement result
            result = RefinementResult(
                cluster_number=cluster_number,
                r_work=r_work,
                r_free=r_free,
                refinement_folder=refinement_folder,
                partial_pdb_path=partial_pdb_path,
                phaser_output_dir=phaser_output_dir,
                mode=mode,
                tfz_score=tfz_score,
                is_complete=True,
                process=process
            )

            # Store result in refinement_results dictionary
            self.refinement_results[cluster_number] = result

            # If R-free is below threshold, put result in success queue
            if r_free < self.r_free_threshold:
                logging.success(f"Refinement for cluster {cluster_number} successful with R-free: {r_free:.4f}")
                self.success_queue.put(result)
            else:
                logging.info(f"Refinement for cluster {cluster_number} completed but R-free ({r_free:.4f}) above threshold")

            return result

        except Exception as e:
            logging.error(f"Refinement for cluster {cluster_number} failed: {e}")
            raise

    def check_completed_refinements(self) -> Optional[RefinementResult]:
        # Check for successful refinements first
        try:
            return self.success_queue.get_nowait()
        except Empty:  # Changed from Queue.Empty to Empty
            pass

        # Check completed refinements
        completed = []
        for cluster_number, future in self.refinement_futures.items():
            if future.done():
                completed.append(cluster_number)
                try:
                    result = future.result()
                    self.refinement_results[cluster_number] = result
                except Exception as e:
                    logging.error(f"Refinement for cluster {cluster_number} failed: {e}")

        # Clean up completed futures
        for cluster_number in completed:
            del self.refinement_futures[cluster_number]

        return None

    def cleanup(self):
        self.executor.shutdown(wait=False)

    def terminate_all_refinements(self):
        """Terminate all running refinements"""
        # First terminate all running phenix.refine processes
        for cluster_number, result in self.refinement_results.items():
            if result and result.process and result.process.poll() is None:
                result.process.terminate()
                logging.info(f"Terminated refinement process for cluster {cluster_number}")
        
        # Then cancel futures and clean up
        for cluster_number, future in self.refinement_futures.items():
            if not future.done():
                future.cancel()
                logging.info(f"Cancelled refinement future for cluster {cluster_number}")
        
        # Clear everything
        self.refinement_futures.clear()
        self.refinement_results.clear()
        
        # Empty the success queue
        while True:
            try:
                self.success_queue.get_nowait()
            except Empty:
                break