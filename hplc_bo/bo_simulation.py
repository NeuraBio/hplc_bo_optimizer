"""
Bayesian Optimization simulation module for HPLC method development.

This module provides a realistic simulation of how Bayesian Optimization would perform
if used sequentially for HPLC method development, by matching BO suggestions to
the closest historical runs.
"""

import os
from typing import Any, Dict, List

from hplc_bo.simulation_runner import SimulationStudyRunner
from hplc_bo.validation import ValidationResult


def simulate_bo_performance(
    validation_results: List[ValidationResult],
    n_trials: int = 50,
    output_dir: str = "bo_simulation",
    use_vector_similarity: bool = True,
    vector_cache_file: str = None,
    similarity_metric: str = "cosine",
    validation_file: str = None,
) -> Dict[str, Any]:
    """Simulate how Bayesian Optimization would have performed using historical data.

    This function simulates the sequential BO process by:
    1. Starting with an empty model
    2. Having BO suggest parameters
    3. Finding the closest matching historical run
    4. Feeding that result back to BO
    5. Repeating and tracking convergence

    Args:
        validation_results: List of ValidationResult objects
        n_trials: Number of BO trials to simulate
        output_dir: Directory to save results
        use_vector_similarity: Whether to use vector similarity for matching runs
        vector_cache_file: Path to vector similarity cache file (optional)
        similarity_metric: Distance metric to use for vector similarity ('cosine', 'euclidean', etc.)

    Returns:
        Dictionary with simulation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # If validation_file is provided, use it directly; otherwise, save validation results to a temporary file
    if validation_file is None and validation_results:
        import json
        from dataclasses import asdict

        # Create a temporary directory for validation results
        temp_dir = os.path.join(output_dir, "temp_validation")
        os.makedirs(temp_dir, exist_ok=True)

        # Save validation results to a temporary JSON file
        temp_validation_file = os.path.join(temp_dir, "validation_details.json")

        # Convert validation results to JSON-serializable format
        json_data = []
        for result in validation_results:
            # Convert dataclass to dict, handling potential non-serializable values
            result_dict = {}
            for key, value in asdict(result).items():
                # Handle numpy arrays or other non-serializable types if needed
                result_dict[key] = value
            json_data.append(result_dict)

        # Save to JSON file
        with open(temp_validation_file, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        validation_file = temp_validation_file
        print(f"Saved validation results to temporary file: {validation_file}")

    # Create a simulation study runner using the validation file
    simulation_runner = SimulationStudyRunner(
        validation_file=validation_file,
        n_trials=n_trials,
        use_vector_similarity=use_vector_similarity,
        similarity_cache_file=vector_cache_file,
        similarity_metric=similarity_metric,
    )

    # Run the simulation
    simulation_results = simulation_runner.run_simulation(n_trials=n_trials)

    # Return simulation results
    return simulation_results
