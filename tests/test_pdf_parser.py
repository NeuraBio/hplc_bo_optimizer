import os

import pytest

from hplc_bo.pdf_parser import extract_rt_table_data

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
