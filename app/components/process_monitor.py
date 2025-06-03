"""
Process Monitor Component

This module provides a component to monitor background processes.
"""

import streamlit as st

from app.utils.session import clean_old_processes, get_active_processes, get_recent_processes
from app.utils.ui_helpers import format_time_elapsed, section_header, subsection_header


def render_process_monitor():
    """
    Render the process monitor component.
    """
    # Clean old processes
    clean_old_processes(max_age_hours=24.0)

    # Get active and recent processes
    active_processes = get_active_processes()
    recent_processes = get_recent_processes(limit=5)

    # Display active processes
    if active_processes:
        section_header("Active Processes")

        for process in active_processes:
            with st.expander(f"{process['name']} - {process['status']}", expanded=True):
                st.caption(process["description"])

                # Show progress bar
                st.progress(process["progress"])

                # Show elapsed time
                elapsed = format_time_elapsed(process["start_time"])
                st.caption(f"Running for: {elapsed}")

    # Display recent completed processes
    if recent_processes:
        subsection_header("Recent Processes")

        for process in recent_processes:
            # Skip active processes as they're already shown above
            if process["status"] == "running":
                continue

            # Status is determined directly from the process dictionary
            # Status color is now determined directly in the UI elements where needed

            with st.expander(f"{process['name']} - {process['status']}", expanded=False):
                st.caption(process["description"])

                # Show completion time
                if process["end_time"]:
                    duration = format_time_elapsed(process["start_time"])
                    st.caption(f"Duration: {duration}")

                # Show error if failed
                if process["status"] == "failed" and process["error"]:
                    st.error(process["error"])

                # Show result if available
                if process["result"]:
                    if isinstance(process["result"], dict) and "message" in process["result"]:
                        st.success(process["result"]["message"])
                    elif isinstance(process["result"], str):
                        st.success(process["result"])
