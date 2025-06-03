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
        st.session_state.processes = []


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
    Register a new background process in session state.

    Args:
        name: Process name
        description: Process description

    Returns:
        Process ID
    """
    process_id = str(uuid.uuid4())
    process = {
        "id": process_id,
        "name": name,
        "description": description,
        "status": "running",
        "start_time": time.time(),
        "end_time": None,
        "progress": 0.0,
        "result": None,
        "error": None,
    }

    if "processes" not in st.session_state:
        st.session_state.processes = []

    st.session_state.processes.append(process)
    return process_id


def update_process(
    process_id: str,
    progress: float = None,
    status: str = None,
    result: Any = None,
    error: str = None,
):
    """
    Update an existing process in session state.

    Args:
        process_id: Process ID
        progress: Optional progress value (0.0 to 1.0)
        status: Optional status update
        result: Optional result data
        error: Optional error message
    """
    if "processes" not in st.session_state:
        return

    for process in st.session_state.processes:
        if process["id"] == process_id:
            if progress is not None:
                process["progress"] = max(0.0, min(1.0, progress))
            if status is not None:
                process["status"] = status
            if result is not None:
                process["result"] = result
            if error is not None:
                process["error"] = error
            break


def complete_process(process_id: str, result: Any = None):
    """
    Mark a process as completed.

    Args:
        process_id: Process ID
        result: Optional result data
    """
    if "processes" not in st.session_state:
        return

    for process in st.session_state.processes:
        if process["id"] == process_id:
            process["status"] = "completed"
            process["end_time"] = time.time()
            process["progress"] = 1.0
            if result is not None:
                process["result"] = result
            break


def fail_process(process_id: str, error: str):
    """
    Mark a process as failed.

    Args:
        process_id: Process ID
        error: Error message
    """
    if "processes" not in st.session_state:
        return

    for process in st.session_state.processes:
        if process["id"] == process_id:
            process["status"] = "failed"
            process["end_time"] = time.time()
            process["error"] = error
            break


def get_process(process_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a process by ID.

    Args:
        process_id: Process ID

    Returns:
        Process dictionary or None if not found
    """
    if "processes" not in st.session_state:
        return None

    for process in st.session_state.processes:
        if process["id"] == process_id:
            return process

    return None


def get_active_processes() -> List[Dict[str, Any]]:
    """
    Get all active (running) processes.

    Returns:
        List of active process dictionaries
    """
    if "processes" not in st.session_state:
        return []

    return [p for p in st.session_state.processes if p["status"] == "running"]


def get_recent_processes(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent processes, sorted by start time (newest first).

    Args:
        limit: Maximum number of processes to return

    Returns:
        List of recent process dictionaries
    """
    if "processes" not in st.session_state:
        return []

    sorted_processes = sorted(
        st.session_state.processes, key=lambda p: p.get("start_time", 0), reverse=True
    )

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
