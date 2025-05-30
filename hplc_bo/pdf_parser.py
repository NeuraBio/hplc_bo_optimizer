import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

# Suppress the annoying CropBox warnings from pdfplumber
warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")

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

# Keywords for finding column temperature
COLUMN_TEMP_KEYWORDS = [
    "Column Temperature",
    "Column Temp",
    "Column Oven",
    "Oven Temperature",
    "Temperature",
]

# Keywords for finding solvent names
SOLVENT_A_KEYWORDS = ["Solvent A", "Mobile Phase A", "Buffer A", "Eluent A"]
SOLVENT_B_KEYWORDS = ["Solvent B", "Mobile Phase B", "Buffer B", "Eluent B"]

# Keywords for finding pH values
pH_KEYWORDS = ["pH", "Buffer pH", "Mobile Phase pH", "Eluent pH"]


@dataclass
class ChromatographyConditions:
    """Class to store chromatography conditions extracted from a PDF."""

    column_temperature: float = None  # in °C
    solvent_a_name: str = None
    solvent_b_name: str = None
    gradient_table: list = None  # List of dicts with time, %A, %B, etc.
    flow_rate: float = None  # in mL/min
    pH: float = None  # pH of the mobile phase
    injection_id: int = None
    result_id: int = None
    sample_set_id: int = None


def extract_rt_table_data(pdf_path: str, verbose: bool = True) -> Optional[List[Dict[str, Any]]]:
    """Extracts the main results table (RT, Tailing Factor, Plate Count, etc.) from a PDF.

    Args:
        pdf_path: The full path to the PDF file.

    Returns:
        A list of dictionaries, where each dictionary represents a row in the table,
        or None if the table is not found.
    """
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

                                # Check if this row should be included based on Area
                                # Per chemist requirement: Ignore rows with no Area or area under 500,000
                                area_found = False
                                for header in headers:
                                    if "area" in header.lower():
                                        try:
                                            area_value = (
                                                float(row_dict[header]) if row_dict[header] else 0
                                            )
                                            if (
                                                area_value < 500000
                                            ):  # Filter out rows with area < 500,000
                                                if verbose:
                                                    print(
                                                        f"Skipping row with low area value: {area_value}"
                                                    )
                                                break
                                            area_found = True
                                        except (ValueError, TypeError):
                                            # If area can't be converted to float, skip this row
                                            if verbose:
                                                print(
                                                    f"Skipping row with invalid area value: {row_dict[header]}"
                                                )
                                            break

                                # Only add rows with valid area values
                                if area_found:
                                    table_data.append(row_dict)
                            else:
                                if verbose:
                                    print(
                                        f"Skipping malformed row in RT table on page {page_num + 1}, table {table_idx + 1}: {row}"
                                    )
                        return table_data  # Assuming only one such table per PDF for now

    except Exception as e:
        if verbose:
            print(f"Error processing PDF {pdf_path}: {e}")
        return None
    return None  # Table not found


def calculate_width_at_base(
    rt: float, plate_count: Optional[float] = None, height: Optional[float] = None
) -> float:
    """
    Calculate peak width at base using the plate count formula and/or peak height.

    The relationship between plate count (N), retention time (RT), and peak width (W) is:
    N = 16 * (RT/W)²

    So we can solve for W:
    W = RT * 4 / sqrt(N)

    Args:
        rt: Retention time in minutes
        plate_count: Theoretical plate count (N)
        height: Peak height (optional, used as a fallback)

    Returns:
        Estimated peak width at base in minutes
    """
    # Default width as a percentage of RT if we can't calculate
    default_width = rt * 0.05

    try:
        if plate_count is not None and plate_count > 0:
            # Calculate width from plate count
            width = rt * 4 / math.sqrt(plate_count)
            return max(width, 0.001)  # Ensure positive width
        elif height is not None and height > 0:
            # Rough approximation based on peak height
            # This is a heuristic and may need adjustment
            width = rt / (height**0.5) * 0.1
            return max(width, 0.001)  # Ensure positive width
        else:
            return default_width
    except (ValueError, TypeError, ZeroDivisionError):
        return default_width


def extract_usp_score_data(
    pdf_path: str, verbose: bool = True
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract retention times, peak widths, and tailing factors from a PDF.

    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to print detailed progress messages

    Returns:
        Tuple of (retention_times, peak_widths, tailing_factors)
    """
    # Extract the RT table data
    rt_data = extract_rt_table_data(pdf_path, verbose=verbose)

    if not rt_data:
        if verbose:
            print(f"No RT table data found in {pdf_path}")
        return [], [], []

    # Get the headers from the first row
    headers = list(rt_data[0].keys())
    if verbose:
        print(f"Available headers in PDF: {headers}")

    # Map common header variations to standard names
    header_mapping = {
        "rt": ["rt", "retention time", "ret. time"],
        "width": ["width", "peak width", "width at base", "width at tangent", "w"],
        "tailing": ["tailing", "tailing factor", "tf", "asymmetry", "as"],
        "plate_count": ["plate count", "plates", "n", "theoretical plates"],
        "height": ["height", "peak height", "h"],
    }

    # Find the actual header names in the data
    rt_header = None
    width_header = None
    tailing_header = None
    plate_count_header = None
    height_header = None

    for header in headers:
        header_lower = header.lower()

        # Check for RT header
        if any(name in header_lower for name in header_mapping["rt"]):
            rt_header = header
            if verbose:
                print(f"Found RT header: {rt_header}")

        # Check for width header
        elif any(name in header_lower for name in header_mapping["width"]):
            width_header = header
            if verbose:
                print(f"Found Width header: {width_header}")

        # Check for tailing header
        elif any(name in header_lower for name in header_mapping["tailing"]):
            tailing_header = header
            if verbose:
                print(f"Found Tailing header: {tailing_header}")

        # Check for plate count header
        elif any(name in header_lower for name in header_mapping["plate_count"]):
            plate_count_header = header
            if verbose:
                print(f"Found Plate Count header: {plate_count_header}")

        # Check for height header
        elif any(name in header_lower for name in header_mapping["height"]):
            height_header = header
            if verbose:
                print(f"Found Height header: {height_header}")

    # Extract the data
    rt_list = []
    peak_widths = []
    tailing_factors = []

    for row in rt_data:
        # Extract RT (required)
        if rt_header and rt_header in row and row[rt_header]:
            try:
                rt = float(row[rt_header])
                rt_list.append(rt)

                # Extract or calculate peak width
                width = None

                # Try to get width from the data
                if width_header and width_header in row and row[width_header]:
                    try:
                        width = float(row[width_header])
                        if width <= 0.001:  # Very small or zero width
                            if verbose:
                                print(
                                    f"Warning: Replacing zero/small width value with default: {width}"
                                )
                            width = rt * 0.05  # Default width is 5% of RT
                    except (ValueError, TypeError):
                        width = rt * 0.05  # Default width is 5% of RT

                # If width is not available, try to calculate from plate count
                if (
                    (width is None or width <= 0.001)
                    and plate_count_header
                    and plate_count_header in row
                    and row[plate_count_header]
                ):
                    try:
                        plate_count = float(row[plate_count_header])
                        if plate_count > 0:
                            width = calculate_width_at_base(rt, plate_count)
                    except (ValueError, TypeError):
                        width = rt * 0.05  # Default width is 5% of RT

                # If still no width, try to estimate from height
                if (
                    (width is None or width <= 0.001)
                    and height_header
                    and height_header in row
                    and row[height_header]
                ):
                    try:
                        height = float(row[height_header])
                        width = calculate_width_at_base(rt, None, height)
                    except (ValueError, TypeError):
                        width = rt * 0.05  # Default width is 5% of RT

                # If all else fails, use default width
                if width is None or width <= 0.001:
                    if verbose:
                        print(f"Using default width for RT {rt}")
                    width = rt * 0.05  # Default width is 5% of RT

                peak_widths.append(width)

                # Extract tailing factor
                tailing = None
                if tailing_header and tailing_header in row and row[tailing_header]:
                    try:
                        tailing = float(row[tailing_header])
                        if tailing <= 0.1 or tailing > 10:  # Invalid tailing value
                            if verbose:
                                print(f"Using default tailing for RT {rt}")
                            tailing = 1.0  # Default tailing factor
                    except (ValueError, TypeError):
                        if verbose:
                            print(f"Using default tailing for RT {rt}")
                        tailing = 1.0  # Default tailing factor
                else:
                    if verbose:
                        print(f"Using default tailing for RT {rt}")
                    tailing = 1.0  # Default tailing factor

                tailing_factors.append(tailing)

            except (ValueError, TypeError):
                # Skip invalid RT values
                pass

    return rt_list, peak_widths, tailing_factors


def extract_run_ids(
    pdf_path: str, verbose: bool = True
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract Injection ID, Result ID, and Sample Set ID from a PDF report.

    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to print detailed progress messages

    Returns:
        Tuple of (injection_id, result_id, sample_set_id)
    """
    injection_id = None
    result_id = None
    sample_set_id = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from the first page only, as IDs should be at the top
            if len(pdf.pages) > 0:
                text = pdf.pages[0].extract_text()
                if text:
                    # Extract Injection ID
                    injection_id_match = re.search(r"Injection ID\s+(\d+)", text)
                    if injection_id_match:
                        injection_id = int(injection_id_match.group(1))

                    # Extract Result ID
                    result_id_match = re.search(r"Result ID\s+(\d+)", text)
                    if result_id_match:
                        result_id = int(result_id_match.group(1))

                    # Extract Sample Set ID
                    sample_set_match = re.search(r"Sample Set Id\s+(\d+)", text)
                    if sample_set_match:
                        sample_set_id = int(sample_set_match.group(1))

                    if verbose and (
                        injection_id is not None
                        or result_id is not None
                        or sample_set_id is not None
                    ):
                        print(
                            f"Extracted IDs - Injection: {injection_id}, Result: {result_id}, Sample Set: {sample_set_id}"
                        )

    except Exception as e:
        if verbose:
            print(f"Error extracting run IDs from {pdf_path}: {e}")

    return injection_id, result_id, sample_set_id


def extract_chromatography_conditions(
    pdf_path: str, verbose: bool = True
) -> ChromatographyConditions:
    """
    Extract chromatography conditions from a PDF report.

    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to print detailed progress messages

    Returns:
        ChromatographyConditions object with extracted data
    """
    conditions = ChromatographyConditions()

    # Extract run IDs
    injection_id, result_id, sample_set_id = extract_run_ids(pdf_path, verbose)
    conditions.injection_id = injection_id
    conditions.result_id = result_id
    conditions.sample_set_id = sample_set_id

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            for _, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue

                # Look for column temperature - try multiple patterns
                temp_patterns = [
                    r"[Cc]olumn [Tt]emperature:?\s*(\d+\.?\d*)\s*[°]?C",  # Column Temperature: 25.0°C
                    r"[Tt]arget [Cc]olumn [Tt]emperature\s*(\d+\.?\d*)\s*\([°]?C\)",  # Target Column Temperature 25.0(°C)
                    r"[Tt]arget [Cc]olumn [Tt]emperature\s*(\d+\.?\d*)",  # Target Column Temperature 25.0
                ]

                for pattern in temp_patterns:
                    temp_matches = re.findall(pattern, text)
                    if temp_matches:
                        try:
                            conditions.column_temperature = float(temp_matches[0])
                            if verbose:
                                print(
                                    f"Found column temperature: {conditions.column_temperature}°C"
                                )
                            break
                        except (ValueError, TypeError):
                            pass

                # Look for flow rate - try multiple patterns
                flow_patterns = [
                    r"[Ff]low [Rr]ate:?\s*(\d+\.?\d*)\s*[mM][lL]/[mM][iI][nN]",  # Flow Rate: 0.8 mL/min
                    r"[Ff]low [Rr]ate\s*\([mM][lL]/[mM][iI][nN]\)\s*(\d+\.?\d*)",  # Flow Rate (mL/min) 0.8
                ]

                for pattern in flow_patterns:
                    flow_matches = re.findall(pattern, text)
                    if flow_matches:
                        try:
                            conditions.flow_rate = float(flow_matches[0])
                            if verbose:
                                print(f"Found flow rate: {conditions.flow_rate} mL/min")
                            break
                        except (ValueError, TypeError):
                            pass

                # If flow rate not found in text, try to extract from gradient table
                if conditions.flow_rate is None and conditions.gradient_table:
                    # Get flow rate from first row of gradient table
                    for row in conditions.gradient_table:
                        if "flow_rate" in row and row["flow_rate"] is not None:
                            try:
                                conditions.flow_rate = float(row["flow_rate"])
                                if verbose:
                                    print(
                                        f"Found flow rate from gradient table: {conditions.flow_rate} mL/min"
                                    )
                                break
                            except (ValueError, TypeError):
                                pass

                # Look for solvent names
                solvent_a_matches = re.findall(
                    r"[Ss]olvent [Aa]:?\s*[Nn]ame\s*(.+?)\s*(?:[Ss]olvent|$)", text
                )
                if solvent_a_matches:
                    conditions.solvent_a_name = solvent_a_matches[0].strip()
                    if verbose:
                        print(f"Found Solvent A: Name {conditions.solvent_a_name}")

                # Look for solvent B name
                for keyword in SOLVENT_B_KEYWORDS:
                    solvent_b_pattern = (
                        f"{keyword}[\s:]+([\w\s\d%\/\-\.]+)"  # e.g., "Solvent B: Acetonitrile"
                    )
                    solvent_b_matches = re.findall(solvent_b_pattern, text)
                    if solvent_b_matches:
                        conditions.solvent_b_name = solvent_b_matches[0].strip()
                        if verbose:
                            print(f"Found solvent B: {conditions.solvent_b_name}")
                        break

                # Look for pH value - try multiple patterns
                pH_patterns = [
                    r"pH[\s:]*([\d\.]+)",  # pH: 7.0
                    r"Buffer pH[\s:]*([\d\.]+)",  # Buffer pH: 7.0
                    r"Mobile Phase pH[\s:]*([\d\.]+)",  # Mobile Phase pH: 7.0
                    r"Eluent pH[\s:]*([\d\.]+)",  # Eluent pH: 7.0
                    r"pH[\s:]*(\d+\.\d+)",  # pH: 7.0 (more specific decimal format)
                    r"pH[\s:]*(\d+)",  # pH: 7 (integer format)
                    r"at pH[\s:]*(\d+\.\d+)",  # at pH 7.0
                    r"at pH[\s:]*(\d+)",  # at pH 7
                ]

                for pattern in pH_patterns:
                    pH_matches = re.findall(pattern, text)
                    if pH_matches:
                        try:
                            conditions.pH = float(pH_matches[0])
                            if verbose:
                                print(f"Found pH value: {conditions.pH}")
                            break
                        except (ValueError, TypeError):
                            pass

                # Look for gradient table
                # First try to extract tables from the page
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:  # Need at least header and one data row
                        continue

                    # Check if this looks like a gradient table
                    header_row = table[0]
                    if (
                        header_row
                        and any("time" in str(h).lower() for h in header_row)
                        and any("%" in str(h) for h in header_row)
                    ):
                        if verbose:
                            print(f"Found gradient table with headers: {header_row}")

                        # Extract data rows
                        gradient_data = []
                        initial_flow_rate = None

                        for row in table[1:]:  # Skip header row
                            if not row or all(cell is None or cell == "" for cell in row):
                                continue

                            try:
                                # Find column indices
                                time_idx = next(
                                    (
                                        i
                                        for i, h in enumerate(header_row)
                                        if "time" in str(h).lower()
                                    ),
                                    None,
                                )
                                flow_idx = next(
                                    (
                                        i
                                        for i, h in enumerate(header_row)
                                        if "flow" in str(h).lower()
                                    ),
                                    None,
                                )
                                a_idx = next(
                                    (i for i, h in enumerate(header_row) if "%a" in str(h).lower()),
                                    None,
                                )
                                b_idx = next(
                                    (i for i, h in enumerate(header_row) if "%b" in str(h).lower()),
                                    None,
                                )

                                if time_idx is None:
                                    continue

                                # Extract values
                                gradient_point = {"time": float(row[time_idx])}

                                # Check flow rate
                                current_flow_rate = None
                                if flow_idx is not None and row[flow_idx]:
                                    try:
                                        current_flow_rate = float(row[flow_idx])
                                        gradient_point["flow_rate"] = current_flow_rate

                                        # Set the main flow rate property if not already set
                                        if conditions.flow_rate is None and current_flow_rate > 0:
                                            conditions.flow_rate = current_flow_rate
                                            if verbose:
                                                print(
                                                    f"Found flow rate from gradient table: {conditions.flow_rate} mL/min"
                                                )

                                        # Per chemist requirement: Ignore rows once FlowRate changes
                                        if initial_flow_rate is None:
                                            # Set the initial flow rate from the first row
                                            initial_flow_rate = current_flow_rate
                                        elif current_flow_rate != initial_flow_rate:
                                            # Flow rate has changed, stop processing gradient table
                                            if verbose:
                                                print(
                                                    f"Flow rate changed from {initial_flow_rate} to {current_flow_rate}. Stopping gradient table processing."
                                                )
                                            break
                                    except (ValueError, TypeError):
                                        pass

                                if a_idx is not None and row[a_idx]:
                                    try:
                                        gradient_point["%A"] = float(row[a_idx])
                                    except (ValueError, TypeError):
                                        pass

                                if b_idx is not None and row[b_idx]:
                                    try:
                                        gradient_point["%B"] = float(row[b_idx])
                                    except (ValueError, TypeError):
                                        pass

                                gradient_data.append(gradient_point)
                            except (ValueError, TypeError):
                                # Skip rows with invalid data
                                pass

                        if gradient_data:
                            conditions.gradient_table = gradient_data
                            if verbose:
                                print(f"Extracted {len(gradient_data)} gradient points")
                            break  # Found a valid gradient table, stop looking

    except Exception as e:
        if verbose:
            print(f"Error extracting chromatography conditions: {e}")

    return conditions


def extract_comprehensive_data(
    pdf_path: str, verbose: bool = True
) -> Tuple[List[float], List[float], List[float], ChromatographyConditions]:
    """
    Extract all data needed for HPLC method optimization from a PDF report.

    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to print detailed progress messages

    Returns:
        Tuple of (retention times, peak widths, tailing factors, chromatography conditions)
    """
    # Extract retention times, peak widths, and tailing factors
    rt_list, peak_widths, tailing_factors = extract_usp_score_data(pdf_path, verbose=verbose)

    # Extract chromatography conditions
    conditions = extract_chromatography_conditions(pdf_path, verbose=verbose)

    # Print a summary of what we found
    if verbose:
        print(f"Extracted {len(rt_list)} peaks with RT, width, and tailing data")
        print("Recalculated peak widths using plate count/height data")

    return rt_list, peak_widths, tailing_factors, conditions
