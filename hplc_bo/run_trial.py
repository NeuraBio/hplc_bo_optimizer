import argparse

from hplc_bo.study_runner import StudyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_lab", required=True, help="Client or lab name")
    parser.add_argument("--experiment", required=True, help="Experiment name")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # suggest
    subparsers.add_parser("suggest", help="Suggest a new trial")

    # report
    report_parser = subparsers.add_parser("report", help="Report results for a given trial")
    report_parser.add_argument("--trial_id", type=int, required=True, help="Trial ID to update")
    report_parser.add_argument("--rt_file", required=True, help="CSV with retention times")
    report_parser.add_argument("--gradient_file", help="Optional gradient override")

    # export
    subparsers.add_parser("export", help="Export all trial history to CSV and plot")

    # run_historical
    hist_parser = subparsers.add_parser("run_historical", help="Run BO with historical data")
    hist_parser.add_argument("--rt_file", required=True, help="CSV with retention times")
    hist_parser.add_argument("--gradient_file", required=True, help="CSV with gradient values")

    args = parser.parse_args()
    runner = StudyRunner(client_lab=args.client_lab, experiment=args.experiment)

    if args.command == "suggest":
        runner.suggest()

    elif args.command == "report":
        runner.report_result(trial_id=args.trial_id, rt_csv_path=args.rt_file)

    elif args.command == "export":
        runner.export_results()

    elif args.command == "run_historical":
        print(f"[Placeholder] Will load and process: {args.csv}")
        # runner.run_historical_from_csv(args.csv)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
