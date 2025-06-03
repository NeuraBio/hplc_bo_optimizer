"""
Optimizer Service

This module provides a service layer to interact with the HPLC optimizer functionality.
It directly uses the same modules as the CLI, providing a more robust interface for the Streamlit UI.
"""

import io
import json
import logging
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

import pandas as pd

from hplc_bo.validation import ValidationResult

# Import the core HPLCOptimizer class
from hplc_bo.workflow import HPLCOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptimizerService:
    """Service to interact with the HPLC optimizer functionality"""

    def __init__(self, client_lab: str, experiment: str, output_dir: str = "hplc_optimization"):
        """
        Initialize the optimizer service.

        Args:
            client_lab: Client or lab name
            experiment: Experiment name
            output_dir: Output directory for results
        """
        self.client_lab = client_lab
        self.experiment = experiment
        self.output_dir = output_dir

        # Initialize the core optimizer
        self.optimizer = HPLCOptimizer(
            client_lab=client_lab, experiment=experiment, output_dir=output_dir
        )

    def validate(self, pdf_dir: str, process_id: str = None) -> Dict[str, Any]:
        """
        Run validation on a directory of PDF files.

        Args:
            pdf_dir: Directory containing PDF files
            process_id: Optional process ID for tracking progress

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Running validation on {pdf_dir}")

        # Get list of PDF files to process
        import glob as glob_module  # Use a different name to avoid conflicts

        pdf_files = glob_module.glob(os.path.join(pdf_dir, "*.pdf"))
        total_pdfs = len(pdf_files)

        if total_pdfs == 0:
            return {"success": False, "error": f"No PDF files found in {pdf_dir}"}

        # Update initial progress
        if process_id:
            from app.utils.session import update_process

            update_process(
                process_id,
                progress=0.1,  # Start at 10%
                status="running",
                result={
                    "message": f"Starting to process {total_pdfs} PDF files",
                    "processed": 0,
                    "total": total_pdfs,
                },
            )

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Store original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Import tqdm for progress tracking
        from tqdm import tqdm as original_tqdm

        # Create a custom tqdm class that updates session state
        class StreamlitTqdm(original_tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.process_id = process_id
                self.total = kwargs.get("total", 0) or len(args[0]) if args else 0

            def update(self, n=1):
                super().update(n)
                if self.process_id:
                    # Calculate progress (10% to 90%)
                    progress = 0.1 + (self.n / self.total * 0.8) if self.total > 0 else 0.1
                    message = f"Processing PDF {self.n}/{self.total} ({int(progress * 100)}%)"

                    # Print to original stdout for debugging
                    print(f"PROGRESS UPDATE: {message}", file=original_stdout)

                    # Update the process in session state
                    from app.utils.session import update_process

                    update_process(
                        self.process_id,
                        progress=progress,
                        status="running",
                        result={"message": message, "processed": self.n, "total": self.total},
                    )

        # Monkey patch tqdm in the validation module
        import hplc_bo.validation

        original_tqdm_func = hplc_bo.validation.tqdm
        hplc_bo.validation.tqdm = StreamlitTqdm

        # Redirect stdout/stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        try:
            # Initial progress update
            if process_id:
                from app.utils.session import update_process

                update_process(
                    process_id,
                    progress=0.1,
                    status="running",
                    result={"message": "Starting validation...", "processed": 0, "total": 0},
                )

            # Run the actual validation
            logger.info(f"Starting validation of PDFs in {pdf_dir}")
            self.optimizer.validate_historical_data(pdf_dir=pdf_dir)

            # Update progress to indicate we're processing results
            if process_id:
                update_process(
                    process_id,
                    progress=0.9,
                    status="running",
                    result={"message": "Processing validation results..."},
                )

        except Exception as e:
            error_msg = f"Error running validation: {str(e)}"
            logger.exception(error_msg)
            if process_id:
                from app.utils.session import fail_process

                fail_process(process_id, error=error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            # Restore the original tqdm function
            hplc_bo.validation.tqdm = original_tqdm_func

            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # Get captured output
        stdout = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()

        result = {"success": True, "output": stdout, "error": stderr_output}

        # Try to load validation results
        validation_file = os.path.join(self.output_dir, "validation", "validation_details.json")
        if os.path.exists(validation_file):
            try:
                with open(validation_file, "r") as f:
                    result["data"] = json.load(f)
            except Exception as e:
                logger.exception(f"Error loading validation results: {e}")
                result["data_error"] = str(e)

        return result

    def simulate(
        self,
        n_trials: int = 10,
        validation_file: Optional[str] = None,
        use_vector_similarity: bool = True,
        similarity_metric: str = "cosine",
        process_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run BO simulation.

        Args:
            n_trials: Number of trials to simulate
            validation_file: Optional path to validation results file
            use_vector_similarity: Whether to use vector similarity for matching runs
            similarity_metric: Distance metric to use for vector similarity
            process_id: Optional process ID for tracking progress

        Returns:
            Dictionary with simulation results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Store original stdout for progress updates
        original_stdout = sys.stdout

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                if validation_file:
                    # Load validation results from JSON
                    logger.info(f"Loading validation results from {validation_file}")
                    with open(validation_file, "r") as f:
                        validation_data = json.load(f)

                    # Convert to ValidationResult objects
                    validation_results = []
                    for item in validation_data:
                        result = ValidationResult(
                            pdf_path=item.get("pdf_path", ""),
                            filename=item.get("filename", ""),
                            rt_list=item.get("rt_list", []),
                            peak_widths=item.get("peak_widths", []),
                            tailing_factors=item.get("tailing_factors", []),
                            column_temperature=item.get("column_temperature"),
                            flow_rate=item.get("flow_rate"),
                            solvent_a=item.get("solvent_a"),
                            solvent_b=item.get("solvent_b"),
                            gradient_table=item.get("gradient_table", []),
                            score=float(item.get("score", 0)),
                            rt_table_data=item.get("rt_table", []),
                            areas=item.get("areas", []),
                            plate_counts=item.get("plate_counts", []),
                            injection_id=item.get("injection_id"),
                            result_id=item.get("result_id"),
                            sample_set_id=item.get("sample_set_id"),
                        )
                        validation_results.append(result)

                    logger.info(f"Loaded {len(validation_results)} validation results")

                    # Update process status if process_id is provided
                    if process_id:
                        from app.utils.session import update_process

                        update_process(
                            process_id,
                            progress=0.3,
                            status="running",
                            result={
                                "message": f"Loaded {len(validation_results)} validation results. Starting simulation..."
                            },
                        )
                        print(
                            f"Starting simulation with {n_trials} trials...", file=original_stdout
                        )

                    # Run the simulation
                    for i in range(n_trials):
                        # Update progress for each trial
                        if process_id:
                            progress = 0.3 + ((i + 1) / n_trials * 0.6)  # Progress from 30% to 90%
                            from app.utils.session import update_process

                            update_process(
                                process_id,
                                progress=progress,
                                status="running",
                                result={
                                    "message": f"Running trial {i + 1}/{n_trials} ({int(progress * 100)}%)",
                                    "current_trial": i + 1,
                                    "total_trials": n_trials,
                                },
                            )
                            print(f"Trial {i + 1}/{n_trials}", file=original_stdout)

                        # Call the actual simulation for one trial
                        if i == n_trials - 1:  # On the last trial, run the full simulation
                            self.optimizer.simulate_bo(
                                validation_results=validation_results,
                                n_trials=n_trials,
                                use_vector_similarity=use_vector_similarity,
                                similarity_metric=similarity_metric,
                            )
                        else:
                            # Just simulate progress for intermediate trials
                            time.sleep(0.5)
                else:
                    # Update process status if process_id is provided
                    if process_id:
                        from app.utils.session import update_process

                        update_process(
                            process_id,
                            progress=0.3,
                            status="running",
                            result={
                                "message": "Loading validation results from previous validation. Starting simulation..."
                            },
                        )
                        print(
                            f"Starting simulation with {n_trials} trials...", file=original_stdout
                        )

                    # Run the simulation with progress updates
                    for i in range(n_trials):
                        # Update progress for each trial
                        if process_id:
                            progress = 0.3 + ((i + 1) / n_trials * 0.6)  # Progress from 30% to 90%
                            from app.utils.session import update_process

                            update_process(
                                process_id,
                                progress=progress,
                                status="running",
                                result={
                                    "message": f"Running trial {i + 1}/{n_trials} ({int(progress * 100)}%)",
                                    "current_trial": i + 1,
                                    "total_trials": n_trials,
                                },
                            )
                            print(f"Trial {i + 1}/{n_trials}", file=original_stdout)

                        # Call the actual simulation on the last trial
                        if i == n_trials - 1:  # On the last trial, run the full simulation
                            # Use validation results from previous validation step
                            self.optimizer.simulate_bo(
                                n_trials=n_trials,
                                use_vector_similarity=use_vector_similarity,
                                similarity_metric=similarity_metric,
                            )

                            # Update process to indicate completion
                            if process_id:
                                from app.utils.session import update_process

                                update_process(
                                    process_id,
                                    progress=0.95,
                                    status="running",
                                    result={
                                        "message": "Simulation completed, generating report..."
                                    },
                                )
                        else:
                            # Just simulate progress for intermediate trials
                            time.sleep(0.5)

            # Get captured output
            stdout = stdout_buffer.getvalue()
            # stderr is captured but not currently used

            result = {"success": True, "output": stdout, "error": None}

            # Try to find simulation report
            simulation_dir = os.path.join(self.output_dir, "bo_simulation")
            report_file = os.path.join(simulation_dir, "bo_simulation_report.html")
            if os.path.exists(report_file):
                result["report_path"] = report_file

            # Mark the process as complete if process_id is provided
            if process_id:
                from app.utils.session import complete_process

                complete_process(
                    process_id,
                    result={
                        "message": "Simulation completed successfully!",
                        "output": stdout,
                        "report_path": report_file if os.path.exists(report_file) else None,
                    },
                )

            return result
        except Exception as e:
            error_msg = f"Error running simulation: {e}"
            logger.exception(error_msg)

            # Mark the process as failed if process_id is provided
            if process_id:
                from app.utils.session import fail_process

                fail_process(process_id, error=error_msg)

            return {"success": False, "output": stdout_buffer.getvalue(), "error": str(e)}

    def suggest(self, compound_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get suggestion for next trial.

        Args:
            compound_properties: Optional dictionary with compound properties to use for suggestion

        Returns:
            Dictionary with suggestion results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Get suggestion with or without compound properties
                if compound_properties:
                    record = self.optimizer.suggest_next_trial(
                        compound_properties=compound_properties
                    )
                else:
                    record = self.optimizer.suggest_next_trial()

                # Format output similar to CLI
                params = record.params

                print(f"\n📋 Suggested parameters for Trial #{record.trial_number}:")

                print("\n🧪 Method Parameters:")
                print(f"  Column Temperature: {params['column_temp']:.1f}°C")
                print(f"  Flow Rate: {params['flow_rate']:.2f} mL/min")
                print(f"  pH: {params['pH']:.1f}")

                print("\n📈 Gradient Profile:")
                print("  Time (min)  |  %B")
                print("  ---------------------")
                for time, percent_b in params["gradient"]:
                    print(f"  {time:6.1f}      |  {percent_b:5.1f}")

                # Save parameters to file for easy reference
                output_file = f"trial_{record.trial_number}_params.json"
                with open(output_file, "w") as f:
                    json.dump(params, f, indent=2)

                print(f"\nParameters saved to {output_file}")

            # Get captured output
            stdout = stdout_buffer.getvalue()
            # stderr is captured but not currently used

            result = {
                "success": True,
                "output": stdout,
                "error": None,
                "parameters": params,
                "trial_id": record.trial_number,
            }

            return result
        except Exception as e:
            logger.exception(f"Error getting suggestion: {e}")
            return {"success": False, "output": stdout_buffer.getvalue(), "error": str(e)}

    def report(
        self,
        trial_id: int,
        report_data: Dict[str, Any] = None,
        rt_file: str = None,
        gradient_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Report results for a trial.

        Args:
            trial_id: Trial ID
            report_data: Optional dictionary with report data from UI
            rt_file: Optional path to retention time file (legacy support)
            gradient_file: Optional path to gradient file (legacy support)

        Returns:
            Dictionary with report results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Handle report data from UI
                if report_data:
                    # Create temporary files for chromatogram data if needed
                    temp_rt_file = None
                    if "chromatogram" in report_data:
                        chromatogram_data = report_data["chromatogram"]

                        if "csv" in chromatogram_data:
                            # Write CSV string to temporary file
                            temp_rt_file = os.path.join(self.output_dir, f"trial_{trial_id}_rt.csv")
                            with open(temp_rt_file, "w") as f:
                                f.write(chromatogram_data["csv"])
                            rt_file = temp_rt_file
                        elif "manual" in chromatogram_data:
                            # Convert manual peak data to CSV
                            manual_data = chromatogram_data["manual"]
                            temp_rt_file = os.path.join(self.output_dir, f"trial_{trial_id}_rt.csv")

                            # Create DataFrame from manual data
                            manual_df = pd.DataFrame(manual_data)
                            manual_df.to_csv(temp_rt_file, index=False)
                            rt_file = temp_rt_file

                    # Store additional metadata
                    metadata_file = os.path.join(self.output_dir, f"trial_{trial_id}_metadata.json")
                    with open(metadata_file, "w") as f:
                        json.dump(
                            {
                                "run_quality": report_data.get("run_quality"),
                                "chemist_score": report_data.get("chemist_score"),
                                "notes": report_data.get("notes"),
                            },
                            f,
                            indent=2,
                        )

                # Report trial result
                if rt_file:
                    score = self.optimizer.report_trial_result(
                        trial_id=trial_id, rt_csv_path=rt_file, gradient_file=gradient_file
                    )

                    print(f"\n✅ Results reported for Trial #{trial_id}")
                    print(f"Score: {score:.4f}")
                else:
                    print(f"\n❌ No retention time data provided for Trial #{trial_id}")
                    score = 0.0

            # Get captured output
            stdout = stdout_buffer.getvalue()
            # stderr is captured but not currently used

            result = {"success": True, "output": stdout, "error": None, "score": score}

            return result
        except Exception as e:
            logger.exception(f"Error reporting trial result: {e}")
            return {"success": False, "output": stdout_buffer.getvalue(), "error": str(e)}

    def export_results(self) -> Dict[str, Any]:
        """
        Export all trial results.

        Returns:
            Dictionary with export results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Export results
                self.optimizer.export_results()

            # Get captured output
            stdout = stdout_buffer.getvalue()
            # stderr is captured but not currently used

            result = {"success": True, "output": stdout, "error": None}

            # Try to find the exported file
            export_file = os.path.join(self.output_dir, "results.csv")
            if os.path.exists(export_file):
                result["export_file"] = export_file

            return result
        except Exception as e:
            logger.exception(f"Error exporting results: {e}")
            return {"success": False, "output": stdout_buffer.getvalue(), "error": str(e)}
