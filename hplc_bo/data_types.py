"""
Common data types for HPLC optimization.

This module contains common data structures used across the HPLC optimization
modules to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    injection_id: Optional[int] = None  # Injection ID for chronological ordering
    result_id: Optional[int] = None  # Result ID for secondary ordering
    sample_set_id: Optional[int] = None  # Sample Set ID for grouping related runs
    pH: Optional[float] = None  # pH value
