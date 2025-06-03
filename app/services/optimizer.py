"""
Optimizer Service

This module provides a service layer to interact with the HPLC optimizer functionality.
It directly uses the same modules as the CLI, providing a more robust interface for the Streamlit UI.
"""

import io
import json
import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

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

    def validate(self, pdf_dir: str) -> Dict[str, Any]:
        """
        Run validation on historical PDF data.

        Args:
            pdf_dir: Directory containing PDF reports

        Returns:
            Dictionary with validation results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Run validation
                self.optimizer.validate_historical_data(pdf_dir=pdf_dir)

            # Get captured output
            stdout = stdout_buffer.getvalue()
            # stderr is captured but not currently used

            result = {"success": True, "output": stdout, "error": None}

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
        except Exception as e:
            logger.exception(f"Error running validation: {e}")
            return {"success": False, "output": stdout_buffer.getvalue(), "error": str(e)}

    def simulate(
        self,
        n_trials: int = 10,
        validation_file: Optional[str] = None,
        use_vector_similarity: bool = True,
        similarity_metric: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Run BO simulation.

        Args:
            n_trials: Number of trials to simulate
            validation_file: Optional path to validation results file
            use_vector_similarity: Whether to use vector similarity for matching runs
            similarity_metric: Distance metric to use for vector similarity

        Returns:
            Dictionary with simulation results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

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
                    self.optimizer.simulate_bo(
                        validation_results=validation_results,
                        n_trials=n_trials,
                        use_vector_similarity=use_vector_similarity,
                        similarity_metric=similarity_metric,
                    )
                else:
                    # Use validation results from previous validation step
                    self.optimizer.simulate_bo(
                        n_trials=n_trials,
                        use_vector_similarity=use_vector_similarity,
                        similarity_metric=similarity_metric,
                    )

            # Get captured output
            stdout = stdout_buffer.getvalue()
            # stderr is captured but not currently used

            result = {"success": True, "output": stdout, "error": None}

            # Try to find simulation report
            simulation_dir = os.path.join(self.output_dir, "bo_simulation")
            report_file = os.path.join(simulation_dir, "bo_simulation_report.html")
            if os.path.exists(report_file):
                result["report_path"] = report_file

            return result
        except Exception as e:
            logger.exception(f"Error running simulation: {e}")
            return {"success": False, "output": stdout_buffer.getvalue(), "error": str(e)}

    def suggest(self) -> Dict[str, Any]:
        """
        Get suggestion for next trial.

        Returns:
            Dictionary with suggestion results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Get suggestion
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
        self, trial_id: int, rt_file: str, gradient_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Report results for a trial.

        Args:
            trial_id: Trial ID
            rt_file: Path to retention time file
            gradient_file: Optional path to gradient file

        Returns:
            Dictionary with report results
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Redirect stdout and stderr to capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Report trial result
                score = self.optimizer.report_trial_result(
                    trial_id=trial_id, rt_csv_path=rt_file, gradient_file=gradient_file
                )

                print(f"\n✅ Results reported for Trial #{trial_id}")
                print(f"Score: {score:.4f}")

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
