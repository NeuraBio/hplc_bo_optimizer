#!/usr/bin/env python
"""
HPLC Method Validation Runner

This script provides a command-line interface to run the validation process
for HPLC method optimization using historical PDF reports.
"""

import argparse
import os
import pickle
import sys

from hplc_bo.validation import run_validation
from hplc_bo.validation_bo import run_bo_validation


def main():
    parser = argparse.ArgumentParser(
        description="Validate HPLC method optimization using historical PDF reports"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Parser for the 'score' command
    score_parser = subparsers.add_parser(
        "score", help="Score historical PDF reports and generate validation report"
    )
    score_parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF reports")
    score_parser.add_argument(
        "--output_dir", default="validation_results", help="Directory to save results"
    )
    score_parser.add_argument("--chemist_ratings", help="CSV file with chemist ratings (optional)")

    # Parser for the 'bo' command
    bo_parser = subparsers.add_parser(
        "bo", help="Run Bayesian Optimization validation on previously scored PDFs"
    )
    bo_parser.add_argument(
        "--results_file", required=True, help="Pickle file with validation results"
    )
    bo_parser.add_argument(
        "--output_dir", default="bo_validation", help="Directory to save results"
    )
    bo_parser.add_argument(
        "--n_suggestions", type=int, default=10, help="Number of suggestions to generate"
    )

    # Parser for the 'full' command
    full_parser = subparsers.add_parser(
        "full", help="Run both scoring and BO validation in sequence"
    )
    full_parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF reports")
    full_parser.add_argument(
        "--score_output_dir", default="validation_results", help="Directory to save scoring results"
    )
    full_parser.add_argument(
        "--bo_output_dir", default="bo_validation", help="Directory to save BO validation results"
    )
    full_parser.add_argument("--chemist_ratings", help="CSV file with chemist ratings (optional)")
    full_parser.add_argument(
        "--n_suggestions", type=int, default=10, help="Number of suggestions to generate"
    )

    args = parser.parse_args()

    if args.command == "score":
        # Run scoring validation
        html_path = run_validation(args.pdf_dir, args.output_dir, args.chemist_ratings)
        print(f"\nValidation complete! View the report at: {html_path}")

    elif args.command == "bo":
        # Load validation results
        if not os.path.exists(args.results_file):
            print(f"Error: Results file '{args.results_file}' not found.")
            return 1

        try:
            with open(args.results_file, "rb") as f:
                validation_results = pickle.load(f)

            # Run BO validation
            html_path = run_bo_validation(validation_results, args.output_dir, args.n_suggestions)
            print(f"\nBO validation complete! View the report at: {html_path}")

        except Exception as e:
            print(f"Error loading validation results: {e}")
            return 1

    elif args.command == "full":
        # Run scoring validation
        print("\n=== Phase 1: Scoring Validation ===")
        html_path = run_validation(args.pdf_dir, args.score_output_dir, args.chemist_ratings)
        print(f"\nScoring validation complete! View the report at: {html_path}")

        # Load validation results
        results_file = os.path.join(args.score_output_dir, "validation_details.json")
        pickle_file = os.path.join(args.score_output_dir, "validation_results.pkl")

        # Save results to pickle for BO validation
        try:
            import json

            from hplc_bo.validation import ValidationResult

            # Load JSON results
            with open(results_file, "r") as f:
                json_results = json.load(f)

            # Convert to ValidationResult objects
            validation_results = []
            for result_dict in json_results:
                validation_results.append(ValidationResult(**result_dict))

            # Save to pickle
            with open(pickle_file, "wb") as f:
                pickle.dump(validation_results, f)

            print(f"Saved validation results to {pickle_file}")

            # Run BO validation
            print("\n=== Phase 2: Bayesian Optimization Validation ===")
            html_path = run_bo_validation(
                validation_results, args.bo_output_dir, args.n_suggestions
            )
            print(f"\nBO validation complete! View the report at: {html_path}")

        except Exception as e:
            print(f"Error in BO validation phase: {e}")
            return 1

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
