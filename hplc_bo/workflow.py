"""
Unified workflow module for HPLC method optimization.

This module provides a cohesive interface for the entire HPLC optimization workflow:
1. Validation of historical data
2. Simulation of BO performance
3. Human-in-the-loop trials with BO suggestions

It serves as the main entry point for all HPLC optimization tasks.
"""

import os
from typing import Any, Dict, List, Optional

from hplc_bo.bo_simulation import simulate_bo_performance
from hplc_bo.gradient_utils import TrialRecord
from hplc_bo.study_runner import StudyRunner
from hplc_bo.validation import ValidationResult, process_pdf_directory


class HPLCOptimizer:
    """
    Unified workflow manager for HPLC method optimization.

    This class provides a cohesive interface for the entire HPLC optimization process,
    from validation of historical data to running actual BO-guided experiments.
    """

    def __init__(
        self,
        client_lab: str,
        experiment: str,
        output_dir: str = "hplc_optimization",
        storage_path: str = "sqlite:///optuna_storage/hplc_study.db",
    ):
        """
        Initialize the HPLC optimizer.

        Args:
            client_lab: Client or lab name
            experiment: Experiment name
            output_dir: Directory to save results
            storage_path: Path to Optuna storage
        """
        self.client_lab = client_lab
        self.experiment = experiment
        self.output_dir = output_dir
        self.storage_path = storage_path

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize study runner for human-in-the-loop trials
        self.study_runner = StudyRunner(
            client_lab=client_lab,
            experiment=experiment,
            storage_path=storage_path,
        )

        # Store validation results
        self.validation_results = None
        self.simulation_results = None

    def validate_historical_data(
        self,
        pdf_dir: str,
        save_results: bool = True,
    ) -> List[ValidationResult]:
        """
        Validate historical data from PDF reports.

        Args:
            pdf_dir: Directory containing PDF reports
            save_results: Whether to save validation results

        Returns:
            List of ValidationResult objects
        """
        print(f"Processing PDF reports from {pdf_dir}...")
        validation_results = process_pdf_directory(
            pdf_dir=pdf_dir,
            output_dir=os.path.join(self.output_dir, "validation"),
            parallel=True,
            num_workers=12,  # Use 12 workers as mentioned in your previous workflow
        )

        self.validation_results = validation_results

        if save_results:
            # Save validation results as JSON
            validation_dir = os.path.join(self.output_dir, "validation")
            os.makedirs(validation_dir, exist_ok=True)

            validation_data = []
            for result in validation_results:
                # Convert ValidationResult to dict
                result_dict = {
                    "filename": result.filename,
                    "pdf_path": result.pdf_path,
                    "score": result.score,
                    "column_temperature": result.column_temperature,
                    "flow_rate": result.flow_rate,
                    "solvent_a": result.solvent_a,
                    "solvent_b": result.solvent_b,
                    "gradient_table": result.gradient_table,
                    "rt_list": result.rt_list,
                    "peak_widths": result.peak_widths,
                    "tailing_factors": result.tailing_factors,
                    "areas": result.areas,
                    "plate_counts": result.plate_counts,
                    "injection_id": result.injection_id,
                    "result_id": result.result_id,
                    "sample_set_id": result.sample_set_id,
                }
                validation_data.append(result_dict)

            # Save as JSON
            import json

            with open(os.path.join(validation_dir, "validation_details.json"), "w") as f:
                json.dump(validation_data, f, indent=2)

        print(f"Processed {len(validation_results)} PDF reports")

        # Generate HTML report
        from hplc_bo.validation import analyze_validation_results, generate_html_report

        validation_dir = os.path.join(self.output_dir, "validation")
        # Analyze results and generate visualizations
        analyze_validation_results(validation_dir)

        # Generate HTML report
        html_path = generate_html_report(validation_dir)
        if html_path:
            print(f"Generated HTML report at {html_path}")

        return validation_results

    def simulate_bo(
        self,
        validation_results: Optional[List[ValidationResult]] = None,
        n_trials: int = 50,
        use_vector_similarity: bool = True,
        vector_cache_file: Optional[str] = None,
        similarity_metric: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Simulate Bayesian Optimization performance using historical data.

        Args:
            validation_results: List of ValidationResult objects (uses self.validation_results if None)
            n_trials: Number of BO trials to simulate
            use_vector_similarity: Whether to use vector similarity for matching runs
            vector_cache_file: Path to vector similarity cache file (optional, auto-detected if None)
            similarity_metric: Distance metric to use for vector similarity ('cosine', 'euclidean', etc.)

        Returns:
            Dictionary with simulation results
        """
        if validation_results is None:
            validation_results = self.validation_results

        if validation_results is None:
            raise ValueError("No validation results available. Run validate_historical_data first.")

        # Auto-detect the vector similarity cache file if not provided and use_vector_similarity is True
        if use_vector_similarity and vector_cache_file is None:
            validation_dir = os.path.join(self.output_dir, "validation")
            auto_cache_file = os.path.join(validation_dir, "vector_similarity_cache.json")

            if os.path.exists(auto_cache_file):
                print(f"Found vector similarity cache at {auto_cache_file}")
                vector_cache_file = auto_cache_file
            else:
                print(f"No vector similarity cache found at {auto_cache_file}")
                print("Vector similarity will be computed during simulation.")

        print(f"Simulating BO performance with {n_trials} trials...")
        simulation_results = simulate_bo_performance(
            validation_results=validation_results,
            n_trials=n_trials,
            output_dir=os.path.join(self.output_dir, "bo_simulation"),
            use_vector_similarity=use_vector_similarity,
            vector_cache_file=vector_cache_file,
            similarity_metric=similarity_metric,
        )

        self.simulation_results = simulation_results

        # Print summary
        print("\nSimulation Results Summary:")
        print(f"Best score found: {simulation_results['best_score']:.2f}")
        print(f"Manual best score: {simulation_results['manual_best_score']:.2f}")
        print(f"Manual best trial: {simulation_results['manual_best_trial']}")

        # Print convergence metrics
        print("\nConvergence Metrics:")
        for threshold in [80, 90, 95, 99]:
            threshold_key = f"{threshold}%_of_best"
            efficiency_key = f"efficiency_gain_{threshold}%"

            if threshold_key in simulation_results["convergence_data"]:
                bo_trials = simulation_results["convergence_data"][threshold_key]
                efficiency = simulation_results["convergence_data"].get(efficiency_key, "N/A")

                print(f"{threshold}% of best score reached after {bo_trials} trials")
                if isinstance(efficiency, (int, float)):
                    print(
                        f"  Efficiency gain: {efficiency}% fewer trials than manual experimentation"
                    )

        print(f"\nHTML report available at: {simulation_results['html_path']}")

        return simulation_results

    def suggest_next_trial(self) -> TrialRecord:
        """
        Suggest parameters for the next trial using Bayesian Optimization.

        Returns:
            TrialRecord with suggested parameters
        """
        return self.study_runner.suggest()

    def report_trial_result(
        self,
        trial_id: int,
        rt_csv_path: str,
        gradient_file: Optional[str] = None,
    ) -> float:
        """
        Report the result of a trial to the BO system.

        Args:
            trial_id: Trial ID
            rt_csv_path: Path to CSV file with retention times
            gradient_file: Optional path to CSV file with gradient values

        Returns:
            Score for the trial
        """
        return self.study_runner.report_result(
            trial_id=trial_id,
            rt_csv_path=rt_csv_path,
            gradient_file=gradient_file,
        )

    def export_results(self) -> str:
        """
        Export all trial results to CSV and generate plots.

        Returns:
            Path to the exported results
        """
        return self.study_runner.export_results()


# The command-line interface has been moved to hplc_optimize.py
