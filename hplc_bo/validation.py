"""
Validation module for HPLC method optimization.

This module provides functionality to validate the scoring function against real-world data
by processing a directory of PDF reports, extracting data, computing scores, and analyzing
the results.
"""

import glob
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hplc_bo.data_types import ValidationResult
from hplc_bo.gradient_utils import compute_score_usp
from hplc_bo.pdf_parser import extract_comprehensive_data, extract_rt_table_data

# ValidationResult is now imported from data_types.py


def process_single_pdf(args) -> Union[ValidationResult, None]:
    """
    Process a single PDF file and return a ValidationResult.

    Args:
        args: Tuple containing (pdf_path, chemist_ratings)

    Returns:
        ValidationResult object or None if processing fails
    """
    pdf_path, chemist_ratings = args
    filename = os.path.basename(pdf_path)

    try:
        # Extract data from PDF with verbose=False to reduce output during batch processing
        rt_list, peak_widths, tailing_factors, conditions = extract_comprehensive_data(
            pdf_path, verbose=False
        )

        # Get the original RT table data for additional columns
        rt_table_data = extract_rt_table_data(pdf_path, verbose=False)

        # Extract areas and plate counts from RT table data if available
        areas = []
        plate_counts = []

        if rt_table_data:
            for row in rt_table_data:
                # Try to find area column
                area_value = None
                for header in row.keys():
                    if "area" in header.lower() and "perarea" not in header.lower():
                        try:
                            area_value = float(row[header]) if row[header] else None
                            break
                        except (ValueError, TypeError):
                            pass
                areas.append(area_value)

                # Try to find plate count column
                plate_count = None
                for header in row.keys():
                    if any(term in header.lower() for term in ["plate", "plates", "platecount"]):
                        try:
                            plate_count = float(row[header]) if row[header] else None
                            break
                        except (ValueError, TypeError):
                            pass
                plate_counts.append(plate_count)

        # Compute score with areas and plate counts if available
        score = compute_score_usp(
            rt_list,
            peak_widths,
            tailing_factors,
            areas=areas if areas else None,
            plate_counts=plate_counts if plate_counts else None,
        )

        # Get chemist rating if available
        chemist_rating = None
        if chemist_ratings and filename in chemist_ratings:
            chemist_rating = chemist_ratings[filename]

        # Create result object
        result = ValidationResult(
            pdf_path=pdf_path,
            filename=filename,
            rt_list=rt_list,
            peak_widths=peak_widths,
            tailing_factors=tailing_factors,
            column_temperature=conditions.column_temperature,
            flow_rate=conditions.flow_rate,
            pH=conditions.pH,  # Include pH from chromatography conditions
            solvent_a=conditions.solvent_a_name,
            solvent_b=conditions.solvent_b_name,
            gradient_table=conditions.gradient_table,
            score=score,
            chemist_rating=chemist_rating,
            rt_table_data=rt_table_data,
            areas=areas,
            plate_counts=plate_counts,
            injection_id=conditions.injection_id,
            result_id=conditions.result_id,
            sample_set_id=conditions.sample_set_id,
        )

        return result

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def process_pdf_directory(
    pdf_dir: str,
    output_dir: str = "validation_results",
    chemist_ratings: Optional[Dict[str, float]] = None,
    parallel: bool = True,
    num_workers: Optional[int] = None,
) -> List[ValidationResult]:
    """
    Process all PDFs in a directory in parallel, compute scores, and save results.

    Args:
        pdf_dir: Directory containing PDF reports
        output_dir: Directory to save results
        chemist_ratings: Optional dictionary mapping filenames to chemist ratings
        parallel: If True, process PDFs in parallel (default: True)
        num_workers: Number of worker processes to use (default: 90% of CPU cores)

    Returns:
        List of ValidationResult objects
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs in the directory
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    # Track start time for processing rate calculation
    import time

    start_time = time.time()

    if not pdf_files:
        return []

    # Determine optimal number of workers (use 90% of available cores)
    if num_workers is None:
        worker_count = max(1, int(multiprocessing.cpu_count() * 0.9))
        # Allow override with environment variable
        if "HPLC_NUM_WORKERS" in os.environ:
            try:
                env_workers = int(os.environ["HPLC_NUM_WORKERS"])
                if env_workers > 0:
                    worker_count = env_workers
            except ValueError:
                pass
    else:
        worker_count = num_workers

    if parallel:
        print(f"Using {worker_count} CPU cores for parallel processing")
    else:
        print("Running in sequential mode (parallel=False)")

    # Prepare arguments for each PDF
    args_list = [(pdf_path, chemist_ratings) for pdf_path in pdf_files]

    results = []

    if parallel:
        # Process PDFs in parallel
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(process_single_pdf, args): args[0] for args in args_list
            }

            # Process results as they complete with a progress bar
            for _, future in enumerate(
                tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing PDFs")
            ):
                pdf_path = future_to_pdf[future]
                filename = os.path.basename(pdf_path)

                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    else:
        # Process PDFs sequentially
        for args in tqdm(args_list, desc="Processing PDFs"):
            pdf_path = args[0]
            filename = os.path.basename(pdf_path)
            try:
                result = process_single_pdf(args)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Sort results chronologically by injection_id and result_id
    print("Sorting validation results chronologically...")
    # Filter out results with missing IDs
    valid_results = [r for r in results if r.injection_id is not None and r.result_id is not None]
    invalid_results = [r for r in results if r.injection_id is None or r.result_id is None]

    if invalid_results:
        print(
            f"Warning: {len(invalid_results)} PDFs had missing Injection ID or Result ID and will be placed at the end"
        )

    # Calculate processing rate
    end_time = time.time()
    processing_time = end_time - start_time
    processing_rate = len(pdf_files) / processing_time if processing_time > 0 else 0

    print(f"Processed {len(pdf_files)} PDFs in {processing_time:.2f} seconds")
    print(f"Processing rate: {processing_rate:.2f} PDFs/second")

    # Sort valid results by injection_id first, then by result_id
    sorted_results = sorted(valid_results, key=lambda x: (x.injection_id, x.result_id))

    # Add invalid results at the end
    sorted_results.extend(invalid_results)

    print(f"Sorted {len(sorted_results)} validation results chronologically")

    # Save results sorted by score for the validation report
    # but preserve chronological order in the returned list for BO validation
    save_validation_results(sorted_results, output_dir, sort_by_score=True)

    # Precompute vector similarity data
    precompute_vector_similarity(sorted_results, output_dir)

    return sorted_results


def save_validation_results(
    results: List[ValidationResult], output_dir: str, sort_by_score: bool = True
):
    """
    Save validation results to CSV and JSON files.

    Args:
        results: List[ValidationResult] objects
        output_dir: Directory to save results
        sort_by_score: If True, sort results by score (highest to lowest) for the report
                      If False, maintain the order of the input results (e.g., chronological)
    """
    # Sort results by score (highest to lowest) for the validation report if requested
    if sort_by_score:
        # Create a copy to avoid modifying the original list
        results_to_save = sorted(
            results, key=lambda x: x.score if x.score is not None else float("-inf"), reverse=True
        )
        print(
            f"Sorted {len(results_to_save)} validation results by score (highest to lowest) for reporting"
        )
    else:
        results_to_save = results

    # Create a DataFrame for the summary CSV
    summary_data = []
    for result in results_to_save:
        # Calculate average values for areas and plate counts if available
        avg_area = None
        avg_plate_count = None

        if result.areas and any(a is not None for a in result.areas):
            valid_areas = [a for a in result.areas if a is not None]
            avg_area = sum(valid_areas) / len(valid_areas) if valid_areas else None

        if result.plate_counts and any(p is not None for p in result.plate_counts):
            valid_plate_counts = [p for p in result.plate_counts if p is not None]
            avg_plate_count = (
                sum(valid_plate_counts) / len(valid_plate_counts) if valid_plate_counts else None
            )

        summary_data.append(
            {
                "pdf_path": result.pdf_path,
                "filename": result.filename,
                "num_peaks": len(result.rt_list),
                "score": result.score,
                "chemist_rating": result.chemist_rating,
                "column_temperature": result.column_temperature,
                "flow_rate": result.flow_rate,
                "pH": result.pH,  # Include pH in the summary
                "solvent_a": result.solvent_a,
                "solvent_b": result.solvent_b,
                # Include chronological ordering information
                "injection_id": result.injection_id,
                "result_id": result.result_id,
                "sample_set_id": result.sample_set_id,
                # Include additional data as requested by chemists
                "avg_area": avg_area,
                "avg_plate_count": avg_plate_count,
                "rt_list": result.rt_list,  # Include compute_score inputs
                "peak_widths": result.peak_widths,  # Include compute_score inputs
                "tailing_factors": result.tailing_factors,  # Include compute_score inputs
            }
        )

    df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "validation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")

    # Save detailed results to JSON
    json_path = os.path.join(output_dir, "validation_details.json")
    with open(json_path, "w") as f:
        # Convert dataclass objects to dictionaries
        json_data = []
        for result in results_to_save:
            # Convert numpy arrays to lists if present
            result_dict = asdict(result)
            json_data.append(result_dict)

        json.dump(json_data, f, indent=2)

    print(f"Saved detailed results to {json_path}")

    return json_path

    # Always save the original results (chronological order) for BO validation
    # This ensures that the BO validation has access to the chronologically ordered data
    if sort_by_score and results != results_to_save:
        pkl_path = os.path.join(output_dir, "validation_results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved validation results to {pkl_path} for BO validation")


def precompute_vector_similarity(results: List[ValidationResult], output_dir: str) -> str:
    """
    Precompute vector similarity data for all validation results and save to a JSON file.

    Args:
        results: List of ValidationResult objects
        output_dir: Directory to save the vector similarity cache

    Returns:
        Path to the saved vector similarity cache file
    """
    print("\nPrecomputing vector similarity data...")

    # Import VectorSimilarityEngine here to avoid circular imports
    from hplc_bo.vector_similarity import VectorSimilarityEngine

    # Initialize the vector similarity engine with the output directory
    similarity_engine = VectorSimilarityEngine(validation_dir=output_dir)

    # Filter results to include only those with necessary parameters
    valid_results = []
    for result in results:
        if (
            result.gradient_table is not None
            and result.flow_rate is not None
            and result.column_temperature is not None
            and result.score is not None
            and result.score > -1000000000
        ):  # Skip results with invalid scores
            # Ensure pH is set
            if result.pH is None:
                result.pH = 7.0  # Use neutral pH as default

            # Convert gradient table to the format expected by the vector similarity engine
            # This is done inside the VectorSimilarityEngine class, so we don't need to do it here
            valid_results.append(result)

    # Add all valid results to the engine at once
    similarity_engine.precompute_validation_vectors(valid_results)

    # Save the precomputed vectors to the cache file
    similarity_engine._save_cache()

    print(
        f"Precomputed vector similarity data for {len(similarity_engine.validation_results)} validation results"
    )
    print(f"Saved vector similarity cache to {similarity_engine.cache_file}")

    # Test the similarity search functionality
    if similarity_engine.validation_results:
        print("\nTesting vector similarity search...")
        # Use the parameters from the best result as a test query
        best_result = max(results, key=lambda x: x.score if x.score is not None else float("-inf"))

        if (
            best_result.gradient_table
            and best_result.flow_rate is not None
            and best_result.pH is not None
            and best_result.column_temperature is not None
        ):
            # Convert gradient table to the format expected by the vector similarity engine
            test_gradient = []
            for point in best_result.gradient_table:
                if "time" in point and "%b" in point:
                    test_gradient.append([float(point["time"]), float(point["%b"])])

            if test_gradient:
                test_params = {
                    "gradient": test_gradient,
                    "flow_rate": best_result.flow_rate,
                    "pH": best_result.pH,
                    "column_temp": best_result.column_temperature,
                }

                # Find similar runs
                similar_runs = similarity_engine.find_similar_runs(test_params, top_k=3)

                print("\nTop 3 similar runs to the best result:")
                for i, (run, distance) in enumerate(similar_runs, 1):
                    print(f"{i}. {run.filename} (Score: {run.score:.2f}, Distance: {distance:.4f})")

    return similarity_engine.cache_file


def analyze_validation_results(output_dir: str):
    """
    Analyze validation results and generate visualizations.

    Args:
        output_dir: Directory containing validation results
    """
    # Load summary data
    csv_path = os.path.join(output_dir, "validation_summary.csv")

    # Check if the CSV file exists and has content
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"Warning: No data to analyze in {csv_path}. Skipping analysis.")
        return None

    try:
        df = pd.read_csv(csv_path)

        # Check if DataFrame is empty
        if df.empty:
            print(f"Warning: No data found in {csv_path}. Skipping analysis.")
            return None
    except Exception as e:
        print(f"Error reading CSV file: {e}. Skipping analysis.")
        return None

    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Histogram of scores
    plt.figure(figsize=(10, 6))
    plt.hist(df["score"].dropna(), bins=20, alpha=0.7)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Distribution of Computed Scores")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "score_histogram.png"))
    plt.close()

    # 2. Scatter plot of score vs. chemist rating (if available)
    if "chemist_rating" in df.columns and df["chemist_rating"].notna().any():
        plt.figure(figsize=(10, 6))
        plt.scatter(df["chemist_rating"], df["score"], alpha=0.7)
        plt.xlabel("Chemist Rating")
        plt.ylabel("Computed Score")
        plt.title("Computed Score vs. Chemist Rating")

        # Add correlation coefficient
        mask = df["chemist_rating"].notna() & df["score"].notna()
        corr = df.loc[mask, ["chemist_rating", "score"]].corr().iloc[0, 1]
        plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction")

        # Add trend line
        if mask.sum() > 1:
            z = np.polyfit(df.loc[mask, "chemist_rating"], df.loc[mask, "score"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(
                df.loc[mask, "chemist_rating"].min(), df.loc[mask, "chemist_rating"].max(), 100
            )
            plt.plot(x_range, p(x_range), "r--")

        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "score_vs_rating.png"))
        plt.close()

    # 3. Score vs. number of peaks
    plt.figure(figsize=(10, 6))
    plt.scatter(df["num_peaks"], df["score"], alpha=0.7)
    plt.xlabel("Number of Peaks")
    plt.ylabel("Computed Score")
    plt.title("Score vs. Number of Peaks")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "score_vs_num_peaks.png"))
    plt.close()

    # 4. Scatter plots of score vs. method parameters
    for param in ["column_temperature", "flow_rate", "pH"]:
        if param in df.columns and df[param].notna().any():
            plt.figure(figsize=(10, 6))
            plt.scatter(df[param], df["score"], alpha=0.7)

            # Set appropriate labels based on parameter
            if param == "column_temperature":
                plt.xlabel("Column Temperature (°C)")
            elif param == "flow_rate":
                plt.xlabel("Flow Rate (mL/min)")
            elif param == "pH":
                plt.xlabel("pH")
            else:
                plt.xlabel(param.replace("_", " ").title())

            plt.ylabel("Computed Score")
            plt.title(f"Score vs. {param.replace('_', ' ').title()}")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, f"score_vs_{param}.png"))
            plt.close()

    # 6. Top and bottom scoring methods
    top_n = min(10, len(df))

    # Top scoring methods
    top_df = df.nlargest(top_n, "score")
    plt.figure(figsize=(12, 6))
    plt.barh(top_df["filename"], top_df["score"])
    plt.xlabel("Score")
    plt.ylabel("Filename")
    plt.title(f"Top {top_n} Scoring Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_scoring_methods.png"))
    plt.close()

    # Bottom scoring methods
    bottom_df = df.nsmallest(top_n, "score")
    plt.figure(figsize=(12, 6))
    plt.barh(bottom_df["filename"], bottom_df["score"])
    plt.xlabel("Score")
    plt.ylabel("Filename")
    plt.title(f"Bottom {top_n} Scoring Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bottom_scoring_methods.png"))
    plt.close()

    print(f"Generated analysis plots in {plots_dir}")

    # Return the summary DataFrame for further analysis
    return df


def generate_html_report(output_dir: str):
    """
    Generate an HTML report for easy browsing of validation results.

    Args:
        output_dir: Directory containing validation results

    Returns:
        Path to the generated HTML report, or None if no data available
    """
    # Load summary data
    csv_path = os.path.join(output_dir, "validation_summary.csv")

    # Check if the CSV file exists and has content
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"Warning: No data to generate report from {csv_path}.")

        # Create a simple HTML report indicating no PDFs were found
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HPLC Method Validation Report - No Data</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .warning {{ color: #e74c3c; padding: 15px; background-color: #fadbd8; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>HPLC Method Validation Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="warning">
                <h2>No PDF Data Found</h2>
                <p>No PDF files were found or successfully processed. Please check:</p>
                <ul>
                    <li>The PDF directory path is correct</li>
                    <li>The PDF directory is accessible from the Docker container</li>
                    <li>The PDF files are in the expected format</li>
                </ul>
                <p>If using Docker, make sure the PDF directory is properly mounted in the container.</p>
            </div>
        </body>
        </html>
        """

        # Write HTML file
        html_path = os.path.join(output_dir, "validation_report.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        print(f"Generated empty report at {html_path}")
        return html_path

    try:
        df = pd.read_csv(csv_path)

        # Check if DataFrame is empty
        if df.empty:
            print(f"Warning: No data found in {csv_path}.")
            return None

        # Sort by score (descending)
        df = df.sort_values("score", ascending=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HPLC Method Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h4 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .plot-container img {{ max-width: 45%; height: auto; border: 1px solid #ddd; }}
            .pdf-link {{ color: #3498db; text-decoration: none; }}
            .pdf-link:hover {{ text-decoration: underline; }}
            button {{ background-color: #3498db; color: white; border: none; padding: 5px 10px; cursor: pointer; border-radius: 3px; }}
            button:hover {{ background-color: #2980b9; }}
            /* Nested table styles */
            td table {{ margin-bottom: 0; }}
            td table th {{ background-color: #e8f4f8; }}
            td table tr:nth-child(even) {{ background-color: #f2f9fc; }}
        </style>
        <script>
            function toggleDetails(id) {{
                var element = document.getElementById(id);
                if (element.style.display === "none") {{
                    element.style.display = "table-row";
                }} else {{
                    element.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <h1>HPLC Method Validation Report</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Number of PDFs Analyzed</td>
                <td>{len(df)}</td>
            </tr>
            <tr>
                <td>Average Score</td>
                <td>{df['score'].mean():.2f}</td>
            </tr>
            <tr>
                <td>Median Score</td>
                <td>{df['score'].median():.2f}</td>
            </tr>
            <tr>
                <td>Min Score</td>
                <td>{df['score'].min():.2f}</td>
            </tr>
            <tr>
                <td>Max Score</td>
                <td>{df['score'].max():.2f}</td>
            </tr>
        </table>
        
        <h2>Analysis Plots</h2>
        <div class="plot-container">
    """

    # Add plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_files = glob.glob(os.path.join(plots_dir, "*.png"))

    for plot_file in plot_files:
        plot_name = os.path.basename(plot_file)
        rel_path = os.path.join("plots", plot_name)
        html_content += f'<img src="{rel_path}" alt="{plot_name}" />\n'

    html_content += """
        </div>
        
        <h2>Results Table</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Filename</th>
                <th>Score</th>
                <th>Injection ID</th>
                <th>Result ID</th>
                <th>Sample Set ID</th>
                <th>Chemist Rating</th>
                <th>Num Peaks</th>
                <th>Flow Rate</th>
                <th>pH</th>
                <th>Column Temp</th>
                <th>Avg Area</th>
                <th>Avg Plate Count</th>
                <th>PDF Link</th>
                <th>Details</th>
            </tr>
    """

    # Add rows for each result
    for i, (_, row) in enumerate(df.iterrows()):
        pdf_path = row["pdf_path"]
        filename = row["filename"]

        # Create a relative path to the PDF
        pdf_rel_path = os.path.relpath(pdf_path, output_dir)

        # Generate a unique ID for the details section
        details_id = f"details_{i}"

        # Format the average area and plate count with appropriate precision
        avg_area = f"{row['avg_area']:,.0f}" if pd.notna(row["avg_area"]) else "N/A"
        avg_plate_count = (
            f"{row['avg_plate_count']:,.0f}" if pd.notna(row["avg_plate_count"]) else "N/A"
        )

        html_content += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{filename}</td>
                <td>{row['score']:.2f}</td>
                <td>{row['injection_id'] if pd.notna(row['injection_id']) else 'N/A'}</td>
                <td>{row['result_id'] if pd.notna(row['result_id']) else 'N/A'}</td>
                <td>{row['sample_set_id'] if pd.notna(row['sample_set_id']) else 'N/A'}</td>
                <td>{row['chemist_rating'] if pd.notna(row['chemist_rating']) else 'N/A'}</td>
                <td>{row['num_peaks']}</td>
                <td>{row['flow_rate'] if pd.notna(row['flow_rate']) else 'N/A'}</td>
                <td>{row['pH'] if pd.notna(row['pH']) else 'N/A'}</td>
                <td>{row['column_temperature'] if pd.notna(row['column_temperature']) else 'N/A'}</td>
                <td>{avg_area}</td>
                <td>{avg_plate_count}</td>
                <td><a href="{pdf_rel_path}" class="pdf-link" target="_blank">View PDF</a></td>
                <td><button onclick="toggleDetails('{details_id}')">Show Details</button></td>
            </tr>
            <tr id="{details_id}" style="display:none;">
                <td colspan="14">
                    <h4>Compute Score Inputs</h4>
                    <table style="width:100%; margin-bottom:10px;">
                        <tr>
                            <th>Peak #</th>
                            <th>RT (min)</th>
                            <th>Peak Width (min)</th>
                            <th>Tailing Factor</th>
                        </tr>
        """

        # Add the compute_score inputs as a nested table
        rt_list = eval(row["rt_list"]) if isinstance(row["rt_list"], str) else row["rt_list"]
        peak_widths = (
            eval(row["peak_widths"]) if isinstance(row["peak_widths"], str) else row["peak_widths"]
        )
        tailing_factors = (
            eval(row["tailing_factors"])
            if isinstance(row["tailing_factors"], str)
            else row["tailing_factors"]
        )

        for j in range(len(rt_list)):
            html_content += f"""
                        <tr>
                            <td>{j + 1}</td>
                            <td>{rt_list[j]:.2f}</td>
                            <td>{peak_widths[j]:.4f}</td>
                            <td>{tailing_factors[j]:.2f}</td>
                        </tr>
            """

        html_content += """
                    </table>
                </td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    # Write HTML file
    html_path = os.path.join(output_dir, "validation_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Generated HTML report at {html_path}")
    return html_path
