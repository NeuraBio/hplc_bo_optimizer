import argparse

from hplc_bo.study_runner import StudyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_lab", required=True, help="Client or lab name")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--interactive", action="store_true", help="Run one ask/tell loop")
    parser.add_argument("--mock", type=int, help="Run N mock trials")
    parser.add_argument("--export", action="store_true", help="Export trial history")
    args = parser.parse_args()

    runner = StudyRunner(client_lab=args.client_lab, experiment=args.experiment)

    if args.interactive:
        runner.run_interactive()
    elif args.mock:
        runner.run_mock_trials(args.mock)
    elif args.export:
        runner.export_results()
    else:
        print("⚠️ Please specify one of --interactive, --mock <N>, or --export")


if __name__ == "__main__":
    main()
