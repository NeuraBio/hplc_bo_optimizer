"""
Session State Management

This module provides functions for managing Streamlit session state.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st


def init_session_state():
    """
    Initialize session state with default values if they don't exist.
    """
    # General app state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    # User preferences
    if "client_lab" not in st.session_state:
        st.session_state.client_lab = "NuraxsDemo"
    if "experiment" not in st.session_state:
        st.session_state.experiment = "HPLC-Optimization"
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = "hplc_optimization"

    # Navigation state
    if "current_view" not in st.session_state:
        st.session_state.current_view = "home"

    # Process tracking
    if "processes" not in st.session_state:
        st.session_state.processes = {}
    elif isinstance(st.session_state.processes, list):
        # Migrate from list to dictionary if needed
        process_list = st.session_state.processes
        st.session_state.processes = {}
        for process in process_list:
            if isinstance(process, dict) and "id" in process:
                st.session_state.processes[process["id"]] = process


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Get a value from session state with a default fallback.

    Args:
        key: Session state key
        default: Default value if key doesn't exist

    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    """
    Set a value in session state.

    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def register_process(name: str, description: str) -> str:
    """
    Register a new process in session state.

    Args:
        name: Process name
        description: Process description

    Returns:
        Process ID
    """
    process_id = str(uuid.uuid4())

    process = {
        "name": name,
        "description": description,
        "status": "running",
        "progress": 0.0,
        "start_time": time.time(),
        "end_time": None,
        "result": None,
        "error": None,
    }

    if "processes" not in st.session_state:
        st.session_state.processes = {}

    st.session_state.processes[process_id] = process
    return process_id


def update_process(
    process_id: str, progress: float, status: str = "running", result: Dict[str, Any] = None
) -> None:
    """
    Update a process status in the session state.

    Args:
        process_id: Process ID
        progress: Progress value (0-1)
        status: Process status (running, done, error)
        result: Optional result data
    """
    if "processes" not in st.session_state:
        st.session_state.processes = {}

    if process_id in st.session_state.processes:
        # Only update progress if it's greater than current progress or status has changed
        current_progress = st.session_state.processes[process_id].get("progress", 0)
        current_status = st.session_state.processes[process_id].get("status", "")

        # Don't allow progress to go to 100% unless status is 'done'
        if progress >= 0.999 and status == "running":
            progress = 0.99

        # Only update progress if it's greater than current or status changed
        if progress > current_progress or status != current_status:
            st.session_state.processes[process_id]["progress"] = progress
            st.session_state.processes[process_id]["status"] = status

        if result is not None:
            st.session_state.processes[process_id]["result"] = result


def complete_process(process_id: str, result: Any = None):
    """
    Mark a process as complete in session state.

    Args:
        process_id: Process ID
        result: Optional result data
    """
    # First ensure the process exists
    if "processes" not in st.session_state or process_id not in st.session_state.processes:
        return

    # Force progress to 1.0 and status to done
    st.session_state.processes[process_id]["progress"] = 1.0
    st.session_state.processes[process_id]["status"] = "done"

    # Add result if provided
    if result is not None:
        st.session_state.processes[process_id]["result"] = result

    # Update end time
    st.session_state.processes[process_id]["end_time"] = time.time()


def fail_process(process_id: str, error: str):
    """
    Mark a process as failed in session state.

    Args:
        process_id: Process ID
        error: Error message
    """
    # First ensure the process exists
    if "processes" not in st.session_state or process_id not in st.session_state.processes:
        return

    # Update status to error
    st.session_state.processes[process_id]["status"] = "error"
    st.session_state.processes[process_id]["end_time"] = time.time()
    st.session_state.processes[process_id]["error"] = error


def get_process(process_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a process from session state by ID.

    Args:
        process_id: Process ID

    Returns:
        Process data or None if not found
    """
    if "processes" not in st.session_state or process_id not in st.session_state.processes:
        return None

    return st.session_state.processes[process_id]


def get_active_processes() -> List[Dict[str, Any]]:
    """
    Get all active processes from session state.

    Returns:
        List of active processes
    """
    if "processes" not in st.session_state:
        return []

    active_processes = []
    for process_id, process in st.session_state.processes.items():
        if process["status"] not in ["done", "error"]:
            # Add the ID to the process dict for convenience
            process_with_id = process.copy()
            process_with_id["id"] = process_id
            active_processes.append(process_with_id)

    return active_processes


def get_recent_processes(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent processes from session state.

    Args:
        limit: Maximum number of processes to return

    Returns:
        List of recent processes, sorted by start time (newest first)
    """
    if "processes" not in st.session_state:
        return []

    # Convert dictionary to list with process_id included
    process_list = []
    for process_id, process in st.session_state.processes.items():
        process_with_id = process.copy()
        process_with_id["id"] = process_id
        process_list.append(process_with_id)

    # Sort by start time (newest first)
    sorted_processes = sorted(process_list, key=lambda p: p.get("start_time", 0), reverse=True)

    return sorted_processes[:limit]


def clean_old_processes(max_age_hours: float = 24.0):
    """
    Remove old completed or failed processes from session state.

    Args:
        max_age_hours: Maximum age in hours to keep processes
    """
    if "processes" not in st.session_state:
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    st.session_state.processes = [
        p
        for p in st.session_state.processes
        if p["status"] == "running"
        or (current_time - p.get("end_time", current_time)) < max_age_seconds
    ]
