import math
import os

import pytest

from hplc_bo.pdf_parser import (
    ChromatographyConditions,
    calculate_width_at_base,
    extract_chromatography_conditions,
    extract_comprehensive_data,
    extract_rt_table_data,
    extract_usp_score_data,
)

# Define the project root if your test data is relative to it
# For now, using the absolute path provided by the user
TEST_PDF_PATH = "/app/AI_4_2603.pdf"


@pytest.fixture
def sample_pdf_path():
    if not os.path.exists(TEST_PDF_PATH):
        pytest.skip(f"Test PDF file not found at {TEST_PDF_PATH}")
    return TEST_PDF_PATH


def test_extract_rt_table_from_ai_4_2603(sample_pdf_path):
    """Test extracting the RT table from the AI_4_2603.pdf file."""
    rt_data = extract_rt_table_data(sample_pdf_path)

    assert rt_data is not None, "extract_rt_table_data should return a list, not None."
    assert isinstance(rt_data, list), "extract_rt_table_data should return a list."
    assert len(rt_data) > 0, "Extracted RT table data should not be empty."

    print("\n--- Sample Extracted RT Table Data (first 2 rows) ---")
    for i, row in enumerate(rt_data[:2]):
        print(f"Row {i + 1}: {row}")
    print("-----------------------------------------------------")

    # Check if all expected headers are present in the first row's keys
    # This assumes the parser correctly uses the table's first row as headers for the dict keys
    if rt_data:
        first_row_keys = rt_data[0].keys()
        # We need to compare against the *actual* headers found in the PDF,
        # which might be slightly different from our initial EXPECTED_RT_TABLE_HEADERS.
        # For now, let's check for a few critical ones we absolutely need.
        critical_headers = ["RT", "TailingFactor", "PlateCount(N)"]
        for header in critical_headers:
            # Check if the header string is a substring of any key,
            # to be more flexible with potential extra characters or slight variations.
            assert any(
                header in key for key in first_row_keys
            ), f"Critical header '{header}' not found in extracted table keys: {list(first_row_keys)}"

    # Further assertions can be added once we confirm the structure:
    # e.g., checking data types, specific values if known, number of columns.
    # For example, check the first row's RT value (assuming it's a string for now)
    # assert isinstance(rt_data[0].get("RT"), str)


def test_extract_usp_score_data(sample_pdf_path):
    """Test extracting USP score data (RT, peak widths, tailing factors) from the PDF."""
    rt_list, peak_widths, tailing_factors = extract_usp_score_data(sample_pdf_path)

    # Check that we got some data
    assert len(rt_list) > 0, "Should extract at least one retention time"

    # All lists should have the same length
    assert (
        len(rt_list) == len(peak_widths) == len(tailing_factors)
    ), "All data lists should have the same length"

    # Check data types
    assert all(isinstance(rt, float) for rt in rt_list), "All retention times should be floats"
    assert all(
        isinstance(width, float) for width in peak_widths
    ), "All peak widths should be floats"
    assert all(
        isinstance(tf, float) for tf in tailing_factors
    ), "All tailing factors should be floats"

    # Print sample data for inspection
    print("\n--- Sample USP Score Data ---")
    print(f"Retention Times: {rt_list[:5]}")
    print(f"Peak Widths: {peak_widths[:5]}")
    print(f"Tailing Factors: {tailing_factors[:5]}")
    print(f"Total peaks found: {len(rt_list)}")
    print("----------------------------")

    # Basic sanity checks on values
    assert all(rt > 0 for rt in rt_list), "All retention times should be positive"
    assert all(width > 0 for width in peak_widths), "All peak widths should be positive"
    assert all(tf > 0 for tf in tailing_factors), "All tailing factors should be positive"

    # Check that we can use this data with compute_score_usp
    # This is just a smoke test to ensure the data types and formats are compatible
    from hplc_bo.gradient_utils import compute_score_usp

    score = compute_score_usp(rt_list, peak_widths, tailing_factors)
    print(f"Computed USP score: {score}")
    assert isinstance(score, float), "compute_score_usp should return a float"


def test_calculate_width_at_base():
    """Test the peak width calculation using plate count and height."""
    # Test with plate count
    rt = 5.0
    plate_count = 10000  # Typical value for a good column
    width = calculate_width_at_base(rt, plate_count)
    expected_width = rt * 4 / math.sqrt(plate_count)
    assert width == expected_width, "Width calculation from plate count failed"

    # Test with height
    height = 1000  # Arbitrary height value
    width = calculate_width_at_base(rt, None, height)
    assert width > 0, "Width calculation from height should be positive"

    # Test with invalid inputs
    width = calculate_width_at_base(rt, -1000)  # Negative plate count
    assert width == rt * 0.05, "Should use default width for invalid plate count"

    width = calculate_width_at_base(rt, None, None)  # No data
    assert width == rt * 0.05, "Should use default width when no data is provided"


def test_extract_chromatography_conditions(sample_pdf_path):
    """Test extracting chromatography conditions from a PDF."""
    conditions = extract_chromatography_conditions(sample_pdf_path)

    # We can't make specific assertions about the values since they depend on the PDF content,
    # but we can check that the function returns the expected type
    assert isinstance(
        conditions, ChromatographyConditions
    ), "Should return a ChromatographyConditions object"

    # Print the extracted conditions for inspection
    print("\n--- Extracted Chromatography Conditions ---")
    print(f"Column Temperature: {conditions.column_temperature}°C")
    print(f"Flow Rate: {conditions.flow_rate} mL/min")
    print(f"Solvent A: {conditions.solvent_a_name}")
    print(f"Solvent B: {conditions.solvent_b_name}")
    if conditions.gradient_table:
        print(f"Gradient Points: {len(conditions.gradient_table)}")
    print("-------------------------------------------")


def test_extract_comprehensive_data(sample_pdf_path):
    """Test extracting all data from a PDF for HPLC method optimization."""
    rt_list, peak_widths, tailing_factors, conditions = extract_comprehensive_data(sample_pdf_path)

    # Check that we got data in the expected format
    assert len(rt_list) > 0, "Should extract at least one retention time"
    assert (
        len(rt_list) == len(peak_widths) == len(tailing_factors)
    ), "All data lists should have the same length"
    assert isinstance(
        conditions, ChromatographyConditions
    ), "Should return a ChromatographyConditions object"

    # Check data types
    assert all(isinstance(rt, float) for rt in rt_list), "All retention times should be floats"
    assert all(
        isinstance(width, float) for width in peak_widths
    ), "All peak widths should be floats"
    assert all(
        isinstance(tf, float) for tf in tailing_factors
    ), "All tailing factors should be floats"

    # Print sample data for inspection
    print("\n--- Comprehensive Data Extraction Results ---")
    print(f"Peaks: {len(rt_list)}")
    print(f"Sample RT: {rt_list[0] if rt_list else 'None'}")
    print(f"Sample Width: {peak_widths[0] if peak_widths else 'None'}")
    print(f"Sample Tailing: {tailing_factors[0] if tailing_factors else 'None'}")
    print(f"Column Temperature: {conditions.column_temperature}°C")
    print(f"Flow Rate: {conditions.flow_rate} mL/min")
    print("---------------------------------------------")

    # Check that we can use this data with compute_score_usp
    from hplc_bo.gradient_utils import compute_score_usp

    score = compute_score_usp(rt_list, peak_widths, tailing_factors)
    print(f"Computed USP score: {score}")
    assert isinstance(score, float), "compute_score_usp should return a float"
