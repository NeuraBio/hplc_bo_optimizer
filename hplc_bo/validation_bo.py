"""
Bayesian Optimization validation module for HPLC method development.

This module provides functionality to retrospectively validate the Bayesian Optimization
approach using historical PDF reports. It simulates how the optimization would have
performed if it had been used to suggest the methods that were actually run.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

from hplc_bo.pdf_parser import ChromatographyConditions
from hplc_bo.validation import ValidationResult


def extract_method_parameters(conditions: ChromatographyConditions) -> Dict[str, Any]:
    """
    Extract method parameters from chromatography conditions.

    Args:
        conditions: ChromatographyConditions object

    Returns:
        Dictionary of method parameters
    """
    params = {
        "column_temperature": conditions.column_temperature,
        "flow_rate": conditions.flow_rate,
    }

    # Extract gradient parameters
    if conditions.gradient_table:
        # Simplify gradient to key points (e.g., initial %B, final %B, slope)
        try:
            initial_b = next(
                (p.get("%B", 0) for p in conditions.gradient_table if p.get("time", 0) == 0), 0
            )
            final_b = conditions.gradient_table[-1].get("%B", 0)
            max_time = conditions.gradient_table[-1].get("time", 0)

            params["initial_b"] = initial_b
            params["final_b"] = final_b

            if max_time > 0:
                params["gradient_slope"] = (final_b - initial_b) / max_time

            # Look for step changes in the gradient
            for i in range(1, len(conditions.gradient_table)):
                prev = conditions.gradient_table[i - 1]
                curr = conditions.gradient_table[i]

                time_diff = curr.get("time", 0) - prev.get("time", 0)
                b_diff = curr.get("%B", 0) - prev.get("%B", 0)

                if time_diff > 0 and abs(b_diff) > 5:  # Significant change
                    step_key = f"step_{i}_time"
                    params[step_key] = curr.get("time", 0)
                    params[f"step_{i}_b"] = curr.get("%B", 0)
        except Exception as e:
            print(f"Error extracting gradient parameters: {e}")

    return params


def create_study_from_historical_data(
    validation_results: List[ValidationResult], output_dir: str = "bo_validation"
) -> Tuple[optuna.Study, pd.DataFrame]:
    """
    Create an Optuna study from historical data.

    Args:
        validation_results: List of ValidationResult objects
        output_dir: Directory to save results

    Returns:
        Tuple of (Optuna study, DataFrame of trials)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a new study
    study = optuna.create_study(direction="maximize")

    # Process each result and add as a trial
    trials_data = []

    for i, result in enumerate(validation_results):
        if result.score is None:
            continue

        # Extract method parameters
        conditions = ChromatographyConditions(
            column_temperature=result.column_temperature,
            flow_rate=result.flow_rate,
            solvent_a_name=result.solvent_a,
            solvent_b_name=result.solvent_b,
            gradient_table=result.gradient_table,
        )

        params = extract_method_parameters(conditions)

        # Create a trial
        trial = optuna.trial.create_trial(
            params=params,
            value=result.score,
            distributions={
                k: optuna.distributions.UniformDistribution(0, 100) for k in params.keys()
            },
            trial_id=i,
        )

        # Add trial to study
        study.add_trial(trial)

        # Record trial data
        trial_data = {
            "trial_id": i,
            "filename": result.filename,
            "score": result.score,
            "num_peaks": len(result.rt_list),
            **params,
        }
        trials_data.append(trial_data)

    # Create DataFrame
    trials_df = pd.DataFrame(trials_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "historical_trials.csv")
    trials_df.to_csv(csv_path, index=False)
    print(f"Saved {len(trials_df)} trials to {csv_path}")

    return study, trials_df


def analyze_bo_performance(
    study: optuna.Study, trials_df: pd.DataFrame, output_dir: str = "bo_validation"
):
    """
    Analyze the performance of Bayesian Optimization on historical data.

    Args:
        study: Optuna study
        trials_df: DataFrame of trials
        output_dir: Directory to save results
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Plot optimization history
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(plots_dir, "optimization_history.png"))

    # 2. Plot parameter importances
    try:
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, "param_importances.png"))
    except Exception as e:
        print(f"Could not generate parameter importance plot: {e}")

    # 3. Plot contour plots for the most important parameters
    param_names = list(trials_df.columns)
    param_names = [
        p for p in param_names if p not in ["trial_id", "filename", "score", "num_peaks"]
    ]

    if len(param_names) >= 2:
        for i in range(min(3, len(param_names))):
            for j in range(i + 1, min(4, len(param_names))):
                try:
                    fig = plot_contour(study, params=[param_names[i], param_names[j]])
                    fig.write_image(
                        os.path.join(plots_dir, f"contour_{param_names[i]}_{param_names[j]}.png")
                    )
                except Exception as e:
                    print(
                        f"Could not generate contour plot for {param_names[i]} vs {param_names[j]}: {e}"
                    )

    # 4. Plot slice plots for each parameter
    for param in param_names:
        try:
            fig = plot_slice(study, params=[param])
            fig.write_image(os.path.join(plots_dir, f"slice_{param}.png"))
        except Exception as e:
            print(f"Could not generate slice plot for {param}: {e}")

    # 5. Custom analysis: Score improvement over time
    plt.figure(figsize=(10, 6))
    best_scores = np.maximum.accumulate(trials_df["score"])
    plt.plot(range(1, len(best_scores) + 1), best_scores)
    plt.xlabel("Number of Trials")
    plt.ylabel("Best Score So Far")
    plt.title("Score Improvement Over Trials")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "score_improvement.png"))
    plt.close()

    # 6. Convergence analysis: How many trials to reach within X% of best score
    best_score = trials_df["score"].max()
    thresholds = [0.9, 0.95, 0.99]

    convergence_data = {}
    for threshold in thresholds:
        target_score = best_score * threshold
        trials_to_threshold = np.argmax(best_scores >= target_score) + 1
        convergence_data[f"{int(threshold * 100)}%_of_best"] = trials_to_threshold

    # Save convergence data
    with open(os.path.join(output_dir, "convergence_analysis.json"), "w") as f:
        json.dump(convergence_data, f, indent=2)

    print(f"Generated BO analysis plots in {plots_dir}")

    # Return convergence data for reporting
    return convergence_data


def simulate_bo_suggestions(
    study: optuna.Study, n_suggestions: int = 10, output_dir: str = "bo_validation"
):
    """
    Simulate what the Bayesian Optimization would suggest next.

    Args:
        study: Optuna study
        n_suggestions: Number of suggestions to generate
        output_dir: Directory to save results

    Returns:
        DataFrame of suggested parameters
    """
    suggestions = []

    # Create a sampler that uses the existing study
    sampler = optuna.samplers.TPESampler(seed=42)

    # Generate suggestions
    for i in range(n_suggestions):
        # Create a trial
        trial = study.ask(sampler=sampler)

        # Get suggested parameters
        params = trial.params

        # Add to suggestions
        suggestion = {"suggestion_id": i, **params}
        suggestions.append(suggestion)

    # Create DataFrame
    suggestions_df = pd.DataFrame(suggestions)

    # Save to CSV
    csv_path = os.path.join(output_dir, "bo_suggestions.csv")
    suggestions_df.to_csv(csv_path, index=False)
    print(f"Saved {len(suggestions_df)} BO suggestions to {csv_path}")

    return suggestions_df


def generate_bo_html_report(
    study: optuna.Study,
    trials_df: pd.DataFrame,
    convergence_data: Dict[str, int],
    suggestions_df: pd.DataFrame,
    output_dir: str = "bo_validation",
):
    """
    Generate an HTML report for Bayesian Optimization validation.

    Args:
        study: Optuna study
        trials_df: DataFrame of historical trials
        convergence_data: Dictionary of convergence metrics
        suggestions_df: DataFrame of suggested parameters
        output_dir: Directory to save results

    Returns:
        Path to the generated HTML report
    """
    # Calculate summary statistics
    best_score = trials_df["score"].max()
    best_trial_id = trials_df.loc[trials_df["score"].idxmax(), "trial_id"]
    best_filename = trials_df.loc[trials_df["score"].idxmax(), "filename"]

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HPLC Method Bayesian Optimization Validation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .plot-container img {{ max-width: 45%; height: auto; border: 1px solid #ddd; }}
            .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>HPLC Method Bayesian Optimization Validation</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Number of Historical Trials</td>
                <td>{len(trials_df)}</td>
            </tr>
            <tr>
                <td>Best Score</td>
                <td>{best_score:.2f} (Trial {best_trial_id}, {best_filename})</td>
            </tr>
    """

    # Add convergence data
    for threshold, trials in convergence_data.items():
        html_content += f"""
            <tr>
                <td>Trials to Reach {threshold} of Best Score</td>
                <td>{trials}</td>
            </tr>
        """

    html_content += """
        </table>
        
        <h2>Optimization Plots</h2>
        <div class="plot-container">
    """

    # Add plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_files = ["optimization_history.png", "param_importances.png", "score_improvement.png"]

    for plot_file in plot_files:
        plot_path = os.path.join(plots_dir, plot_file)
        if os.path.exists(plot_path):
            rel_path = os.path.join("plots", plot_file)
            plot_name = plot_file.replace("_", " ").replace(".png", "").title()
            html_content += f'<img src="{rel_path}" alt="{plot_name}" title="{plot_name}" />\n'

    html_content += """
        </div>
        
        <h3>Parameter Slice Plots</h3>
        <div class="plot-container">
    """

    # Add slice plots
    slice_plots = [
        f for f in os.listdir(plots_dir) if f.startswith("slice_") and f.endswith(".png")
    ]
    for plot_file in slice_plots:
        rel_path = os.path.join("plots", plot_file)
        param_name = plot_file.replace("slice_", "").replace(".png", "")
        html_content += f'<img src="{rel_path}" alt="Slice Plot: {param_name}" title="Slice Plot: {param_name}" />\n'

    html_content += """
        </div>
        
        <h3>Parameter Contour Plots</h3>
        <div class="plot-container">
    """

    # Add contour plots
    contour_plots = [
        f for f in os.listdir(plots_dir) if f.startswith("contour_") and f.endswith(".png")
    ]
    for plot_file in contour_plots:
        rel_path = os.path.join("plots", plot_file)
        params = plot_file.replace("contour_", "").replace(".png", "").replace("_", " vs ")
        html_content += f'<img src="{rel_path}" alt="Contour Plot: {params}" title="Contour Plot: {params}" />\n'

    html_content += """
        </div>
        
        <h2>Historical Trials</h2>
        <p>Top 10 scoring trials:</p>
        <table>
            <tr>
                <th>Trial ID</th>
                <th>Filename</th>
                <th>Score</th>
    """

    # Add parameter columns
    param_columns = [
        col
        for col in trials_df.columns
        if col not in ["trial_id", "filename", "score", "num_peaks"]
    ]
    for param in param_columns:
        html_content += f"<th>{param}</th>\n"

    html_content += """
            </tr>
    """

    # Add rows for top trials
    top_trials = trials_df.nlargest(10, "score")
    for _, row in top_trials.iterrows():
        html_content += f"""
            <tr class="highlight">
                <td>{int(row['trial_id'])}</td>
                <td>{row['filename']}</td>
                <td>{row['score']:.2f}</td>
        """

        for param in param_columns:
            if param in row and pd.notna(row[param]):
                html_content += (
                    f"<td>{row[param]:.2f if isinstance(row[param], float) else row[param]}</td>\n"
                )
            else:
                html_content += "<td>N/A</td>\n"

        html_content += "</tr>\n"

    html_content += """
        </table>
        
        <h2>Suggested Next Trials</h2>
        <p>Based on the historical data, the Bayesian Optimization would suggest these parameter settings for the next trials:</p>
        <table>
            <tr>
                <th>Suggestion ID</th>
    """

    # Add parameter columns for suggestions
    for param in suggestions_df.columns:
        if param != "suggestion_id":
            html_content += f"<th>{param}</th>\n"

    html_content += """
            </tr>
    """

    # Add rows for suggestions
    for _, row in suggestions_df.iterrows():
        html_content += f"""
            <tr>
                <td>{int(row['suggestion_id'])}</td>
        """

        for param in suggestions_df.columns:
            if param != "suggestion_id" and param in row and pd.notna(row[param]):
                html_content += (
                    f"<td>{row[param]:.2f if isinstance(row[param], float) else row[param]}</td>\n"
                )
            elif param != "suggestion_id":
                html_content += "<td>N/A</td>\n"

        html_content += "</tr>\n"

    html_content += """
        </table>
        
        <h2>Conclusion</h2>
        <p>This report shows how Bayesian Optimization would have performed if it had been used to optimize the HPLC method based on historical data.</p>
        <p>The convergence analysis indicates how many trials would have been needed to reach a certain percentage of the best score.</p>
        <p>The suggested next trials represent what the optimization algorithm would recommend trying next, based on the patterns observed in the historical data.</p>
    </body>
    </html>
    """

    # Write HTML file
    html_path = os.path.join(output_dir, "bo_validation_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Generated BO validation HTML report at {html_path}")
    return html_path


def run_bo_validation(
    validation_results: List[ValidationResult],
    output_dir: str = "bo_validation",
    n_suggestions: int = 10,
):
    """
    Run the complete Bayesian Optimization validation workflow.

    Args:
        validation_results: List of ValidationResult objects
        output_dir: Directory to save results
        n_suggestions: Number of suggestions to generate

    Returns:
        Path to the generated HTML report
    """
    # Create Optuna study from historical data
    study, trials_df = create_study_from_historical_data(validation_results, output_dir)

    # Analyze BO performance
    convergence_data = analyze_bo_performance(study, trials_df, output_dir)

    # Simulate BO suggestions
    suggestions_df = simulate_bo_suggestions(study, n_suggestions, output_dir)

    # Generate HTML report
    html_path = generate_bo_html_report(
        study, trials_df, convergence_data, suggestions_df, output_dir
    )

    print(f"BO validation complete. Results saved to {output_dir}")
    print(f"HTML report available at {html_path}")

    return html_path


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="Validate Bayesian Optimization for HPLC method development"
    )
    parser.add_argument("--results_file", required=True, help="Pickle file with validation results")
    parser.add_argument("--output_dir", default="bo_validation", help="Directory to save results")
    parser.add_argument(
        "--n_suggestions", type=int, default=10, help="Number of suggestions to generate"
    )

    args = parser.parse_args()

    # Load validation results
    with open(args.results_file, "rb") as f:
        validation_results = pickle.load(f)

    # Run BO validation
    run_bo_validation(validation_results, args.output_dir, args.n_suggestions)
