#!/usr/bin/env python
"""
HPLC Method Optimization CLI

This script provides a unified command-line interface for the HPLC method optimization workflow,
integrating validation, simulation, and human-in-the-loop trials.
"""

import argparse
import os

from hplc_bo.workflow import HPLCOptimizer


def main():
    parser = argparse.ArgumentParser(
        description="HPLC Method Optimization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate historical data from PDF reports
  python hplc_optimize.py --client_lab NuraxsDemo --experiment HPLC-5 validate --pdf_dir validation_pdfs

  # Simulate BO performance using validation results
  python hplc_optimize.py --client_lab NuraxsDemo --experiment HPLC-5 simulate --n_trials 40

  # Get a suggestion for the next trial
  python hplc_optimize.py --client_lab NuraxsDemo --experiment HPLC-5 suggest

  # Report results for a trial
  python hplc_optimize.py --client_lab NuraxsDemo --experiment HPLC-5 report --trial_id 1 --rt_file results/rt_data.csv

  # Export all trial results
  python hplc_optimize.py --client_lab NuraxsDemo --experiment HPLC-5 export
        """,
    )

    # Global arguments
    parser.add_argument("--client_lab", required=True, help="Client or lab name")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--output_dir", default="hplc_optimization", help="Output directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate historical data from PDF reports"
    )
    validate_parser.add_argument(
        "--pdf_dir", required=True, help="Directory containing PDF reports"
    )

    # simulate command
    simulate_parser = subparsers.add_parser(
        "simulate", help="Simulate BO performance using historical data"
    )
    simulate_parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of trials to simulate"
    )
    simulate_parser.add_argument(
        "--validation_file",
        help="JSON file with validation results (optional, defaults to latest validation results)",
    )
    simulate_parser.add_argument(
        "--vector_similarity",
        action="store_true",
        default=True,
        help="Use vector similarity for matching runs (default: True)",
    )
    simulate_parser.add_argument(
        "--no_vector_similarity",
        action="store_false",
        dest="vector_similarity",
        help="Disable vector similarity and use traditional distance calculation",
    )
    simulate_parser.add_argument(
        "--similarity_metric",
        default="cosine",
        choices=["cosine", "euclidean", "correlation", "manhattan"],
        help="Distance metric to use for vector similarity (default: cosine)",
    )

    # suggest command
    _ = subparsers.add_parser("suggest", help="Suggest parameters for the next trial")

    # report command
    report_parser = subparsers.add_parser("report", help="Report results for a trial")
    report_parser.add_argument(
        "--trial_id", type=int, required=True, help="Trial ID to report results for"
    )
    report_parser.add_argument("--rt_file", required=True, help="CSV file with retention time data")
    report_parser.add_argument(
        "--gradient_file", help="Optional CSV file with gradient data (if different from suggested)"
    )

    # export command
    _ = subparsers.add_parser("export", help="Export all trial results to CSV and generate plots")

    # Parse arguments
    args = parser.parse_args()

    # Initialize optimizer
    optimizer = HPLCOptimizer(
        client_lab=args.client_lab,
        experiment=args.experiment,
        output_dir=args.output_dir,
    )

    # Execute command
    if args.command == "validate":
        optimizer.validate_historical_data(pdf_dir=args.pdf_dir)
        print(
            f"Validation complete. Results saved to {os.path.join(args.output_dir, 'validation')}"
        )

    elif args.command == "simulate":
        if args.validation_file:
            # Load validation results from JSON
            import json

            from hplc_bo.validation import ValidationResult

            print(f"Loading validation results from {args.validation_file}")
            with open(args.validation_file, "r") as f:
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

            print(f"Loaded {len(validation_results)} validation results")
            optimizer.simulate_bo(
                validation_results=validation_results,
                n_trials=args.n_trials,
                use_vector_similarity=args.vector_similarity,
                similarity_metric=args.similarity_metric,
            )
        else:
            # Use validation results from previous validation step
            optimizer.simulate_bo(
                n_trials=args.n_trials,
                use_vector_similarity=args.vector_similarity,
                similarity_metric=args.similarity_metric,
            )

    elif args.command == "suggest":
        record = optimizer.suggest_next_trial()
        print(f"\n📋 Suggested parameters for Trial #{record.trial_number}:")

        # Print parameters in a user-friendly format
        params = record.params

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
        import json

        output_file = f"trial_{record.trial_number}_params.json"
        with open(output_file, "w") as f:
            json.dump(params, f, indent=2)

        print(f"\nParameters saved to {output_file}")
        print("\nRun the method with these parameters and then report the results using:")
        print(
            f"python hplc_optimize.py --client_lab {args.client_lab} --experiment {args.experiment} report --trial_id {record.trial_number} --rt_file YOUR_RESULTS.csv"
        )

    elif args.command == "report":
        score = optimizer.report_trial_result(
            trial_id=args.trial_id,
            rt_csv_path=args.rt_file,
            gradient_file=args.gradient_file,
        )
        print(f"Reported result for Trial #{args.trial_id}: Score = {score:.2f}")
        print("\nTo get the next suggestion, run:")
        print(
            f"python hplc_optimize.py --client_lab {args.client_lab} --experiment {args.experiment} suggest"
        )

    elif args.command == "export":
        export_path = optimizer.export_results()
        print(f"Exported results to {export_path}")
        print("You can view the results in your browser or spreadsheet application.")


if __name__ == "__main__":
    main()
