from typing import Any, Dict, List, Optional

import pdfplumber

# Expected headers for the main results table (RT table)
# Adjust these based on the exact headers in your PDF
EXPECTED_RT_TABLE_HEADERS = [
    "Injection",
    "Name",
    "RT",
    "Area",
    "PerArea",
    "InjVol(uL)",
    "TailingFactor",
    "PlateCount(N)",
    "Width at Tangent (Plate Count)",  # Or similar variations
]

# Headers for the gradient table
EXPECTED_GRADIENT_TABLE_HEADERS = [
    "Time(min)",
    "Flow Rate(mL/min)",
    "%A",
    "%B",
    "%C",
    "%D",
    "Curve",
]


def extract_rt_table_data(pdf_path: str) -> Optional[List[Dict[str, Any]]]:
    """Extracts the main results table (RT, Tailing Factor, Plate Count, etc.) from a PDF.

    Args:
        pdf_path: The full path to the PDF file.

    Returns:
        A list of dictionaries, where each dictionary represents a row in the table,
        or None if the table is not found.
    """
    all_tables_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table:  # Skip empty tables
                        continue

                    # Check if the first row (headers) matches our expected RT table headers
                    # This needs to be robust to slight variations in header names or order
                    header_row = [str(h).strip() for h in table[0]]

                    # A simple check: see if a significant number of expected headers are present
                    # This could be made more sophisticated
                    matched_headers = sum(
                        1 for h_exp in EXPECTED_RT_TABLE_HEADERS if h_exp in header_row
                    )

                    if matched_headers > len(EXPECTED_RT_TABLE_HEADERS) / 2:  # Heuristic
                        # Found the RT table
                        headers = [str(h).strip() for h in table[0]]
                        table_data = []
                        for row in table[1:]:
                            # Ensure row has same number of elements as headers
                            if len(row) == len(headers):
                                row_dict = {headers[i]: val for i, val in enumerate(row)}
                                table_data.append(row_dict)
                            else:
                                print(
                                    f"Skipping malformed row in RT table on page {page_num+1}, table {table_idx+1}: {row}"
                                )
                        return table_data  # Assuming only one such table per PDF for now

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None
    return None  # Table not found


# We will add functions to extract other information like:
# - Gradient Table
# - Target Column Temperature
# - Solvent A/B Names

if __name__ == "__main__":
    # Example usage (replace with an actual PDF path for testing)
    sample_pdf_path = "/Users/umeshdangat/code/github/hplc_bo_optimizer/AI_4_2603.pdf"

    print(f"Attempting to parse: {sample_pdf_path}")
    rt_data = extract_rt_table_data(sample_pdf_path)
    if rt_data:
        print("\nExtracted RT Table Data:")
        for row_idx, row_data in enumerate(rt_data):
            print(f"Row {row_idx + 1}: {row_data}")
    else:
        print("\nRT Table not found or error in parsing.")
