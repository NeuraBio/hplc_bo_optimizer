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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hplc_bo.gradient_utils import compute_score_usp
from hplc_bo.pdf_parser import extract_comprehensive_data, extract_rt_table_data


@dataclass
class ValidationResult:
    """Class to store validation results for a single PDF."""

    pdf_path: str
    filename: str
    rt_list: List[float]
    peak_widths: List[float]
    tailing_factors: List[float]
    column_temperature: Optional[float] = None
    flow_rate: Optional[float] = None
    solvent_a: Optional[str] = None
    solvent_b: Optional[str] = None
    gradient_table: Optional[List[Dict[str, Any]]] = None
    score: Optional[float] = None
    chemist_rating: Optional[float] = None  # If available
    notes: Optional[str] = None
    # Additional RT table data as requested by chemists
    rt_table_data: Optional[List[Dict[str, Any]]] = None  # Store complete RT table data
    areas: Optional[List[float]] = None  # Peak areas
    plate_counts: Optional[List[float]] = None  # Theoretical plate counts


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
            solvent_a=conditions.solvent_a_name,
            solvent_b=conditions.solvent_b_name,
            gradient_table=conditions.gradient_table,
            score=score,
            chemist_rating=chemist_rating,
            rt_table_data=rt_table_data,
            areas=areas,
            plate_counts=plate_counts,
        )

        return result

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def process_pdf_directory(
    pdf_dir: str,
    output_dir: str = "validation_results",
    chemist_ratings: Optional[Dict[str, float]] = None,
) -> List[ValidationResult]:
    """
    Process all PDFs in a directory in parallel, compute scores, and save results.

    Args:
        pdf_dir: Directory containing PDF reports
        output_dir: Directory to save results
        chemist_ratings: Optional dictionary mapping filenames to chemist ratings

    Returns:
        List of ValidationResult objects
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs in the directory
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    if not pdf_files:
        return []

    # Determine optimal number of workers (use 75% of available cores)
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_workers} CPU cores for parallel processing")

    # Prepare arguments for each PDF
    args_list = [(pdf_path, chemist_ratings) for pdf_path in pdf_files]

    results = []

    # Process PDFs in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_pdf = {executor.submit(process_single_pdf, args): args[0] for args in args_list}

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

    # Save results
    save_validation_results(results, output_dir)

    return results


def save_validation_results(results: List[ValidationResult], output_dir: str):
    """
    Save validation results to CSV and JSON files.

    Args:
        results: List of ValidationResult objects
        output_dir: Directory to save results
    """
    # Create a DataFrame for the summary CSV
    summary_data = []
    for result in results:
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
                "solvent_a": result.solvent_a,
                "solvent_b": result.solvent_b,
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
        for result in results:
            # Convert numpy arrays to lists if present
            result_dict = asdict(result)
            json_data.append(result_dict)

        json.dump(json_data, f, indent=2, default=str)

    print(f"Saved detailed results to {json_path}")


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

    # 4. Score vs. flow rate
    if df["flow_rate"].notna().any():
        plt.figure(figsize=(10, 6))
        plt.scatter(df["flow_rate"], df["score"], alpha=0.7)
        plt.xlabel("Flow Rate (mL/min)")
        plt.ylabel("Computed Score")
        plt.title("Score vs. Flow Rate")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "score_vs_flow_rate.png"))
        plt.close()

    # 5. Score vs. column temperature
    if df["column_temperature"].notna().any():
        plt.figure(figsize=(10, 6))
        plt.scatter(df["column_temperature"], df["score"], alpha=0.7)
        plt.xlabel("Column Temperature (°C)")
        plt.ylabel("Computed Score")
        plt.title("Score vs. Column Temperature")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "score_vs_temperature.png"))
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
                <th>Chemist Rating</th>
                <th>Num Peaks</th>
                <th>Flow Rate</th>
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
                <td>{row['chemist_rating'] if pd.notna(row['chemist_rating']) else 'N/A'}</td>
                <td>{row['num_peaks']}</td>
                <td>{row['flow_rate'] if pd.notna(row['flow_rate']) else 'N/A'}</td>
                <td>{row['column_temperature'] if pd.notna(row['column_temperature']) else 'N/A'}</td>
                <td>{avg_area}</td>
                <td>{avg_plate_count}</td>
                <td><a href="{pdf_rel_path}" class="pdf-link" target="_blank">View PDF</a></td>
                <td><button onclick="toggleDetails('{details_id}')">Show Details</button></td>
            </tr>
            <tr id="{details_id}" style="display:none;">
                <td colspan="11">
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


def run_validation(
    pdf_dir: str, output_dir: str = "validation_results", chemist_ratings_file: Optional[str] = None
):
    """
    Run the complete validation workflow.

    Args:
        pdf_dir: Directory containing PDF reports
        output_dir: Directory to save results
        chemist_ratings_file: Optional CSV file with chemist ratings

    Returns:
        Path to the generated HTML report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if PDF directory exists and is accessible
    if not os.path.exists(pdf_dir):
        print(
            f"Warning: PDF directory '{pdf_dir}' does not exist or is not accessible from the container."
        )
        print(
            "If you're running in Docker, make sure the PDF directory is mounted in the container."
        )
        print("For example, add this to your docker-compose.yml:")
        print("    volumes:")
        print(f"      - {pdf_dir}:{pdf_dir}")

        # Create empty results files
        with open(os.path.join(output_dir, "validation_summary.csv"), "w") as f:
            f.write(
                "filename,score,chemist_rating,num_peaks,column_temperature,flow_rate,solvent_a,solvent_b,has_gradient,pdf_path\n"
            )

        with open(os.path.join(output_dir, "validation_details.json"), "w") as f:
            f.write("[]")

        # Create empty pickle file for BO validation
        with open(os.path.join(output_dir, "validation_results.pkl"), "wb") as f:
            pickle.dump([], f)

        # Generate HTML report with error message
        html_path = generate_html_report(output_dir)
        print(f"Created empty report at {html_path}")
        return html_path

    # Load chemist ratings if provided
    chemist_ratings = None
    if chemist_ratings_file and os.path.exists(chemist_ratings_file):
        try:
            ratings_df = pd.read_csv(chemist_ratings_file)
            # Assuming the CSV has columns "filename" and "rating"
            chemist_ratings = dict(zip(ratings_df["filename"], ratings_df["rating"], strict=False))
            print(f"Loaded {len(chemist_ratings)} chemist ratings from {chemist_ratings_file}")
        except Exception as e:
            print(f"Error loading chemist ratings: {e}")

    # Process PDFs
    results = process_pdf_directory(pdf_dir, output_dir, chemist_ratings)

    # Save results to pickle for BO validation
    pickle_path = os.path.join(output_dir, "validation_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved validation results to {pickle_path} for BO validation")

    # Analyze results if we have any
    if results:
        analyze_validation_results(output_dir)

    # Generate HTML report
    html_path = generate_html_report(output_dir)

    print(f"Validation complete. Processed {len(results)} PDFs.")
    print(f"Results saved to {output_dir}")
    print(f"HTML report available at {html_path}")

    return html_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate HPLC method scoring against real-world data"
    )
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF reports")
    parser.add_argument(
        "--output_dir", default="validation_results", help="Directory to save results"
    )
    parser.add_argument("--chemist_ratings", help="CSV file with chemist ratings (optional)")

    args = parser.parse_args()

    run_validation(args.pdf_dir, args.output_dir, args.chemist_ratings)
