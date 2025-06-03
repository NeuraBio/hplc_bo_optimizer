"""
Navigation Component

This module provides the sidebar navigation for the HPLC Method Optimizer Streamlit app.
"""

import streamlit as st

from app.utils.session import get_session_value, set_session_value


def render_sidebar_navigation() -> str:
    """
    Render the sidebar navigation and return the selected view.

    Returns:
        Selected view name
    """
    st.sidebar.title("Navigation")

    # Define navigation options with icons and descriptions
    nav_options = [
        {"id": "home", "label": "🏠 Home", "description": "Dashboard and overview"},
        {"id": "validate", "label": "📊 Validation", "description": "Process historical data"},
        {
            "id": "simulate",
            "label": "🔄 Simulation",
            "description": "Test optimization performance",
        },
        {"id": "suggest", "label": "💡 Suggestion", "description": "Get parameter suggestions"},
        {"id": "report", "label": "📝 Reporting", "description": "Report experimental results"},
        {"id": "status", "label": "📈 Status", "description": "View experiment status"},
        {"id": "advanced", "label": "⚙️ Advanced", "description": "Advanced settings and tools"},
    ]

    # Get current view from session state
    current_view = get_session_value("current_view", "home")

    # Create radio buttons for navigation
    selected_view = st.radio(
        "Select View",
        options=[opt["id"] for opt in nav_options],
        format_func=lambda x: next((opt["label"] for opt in nav_options if opt["id"] == x), x),
        index=[opt["id"] for opt in nav_options].index(current_view),
        key="navigation_radio",
    )

    # Show description for selected view
    selected_description = next(
        (opt["description"] for opt in nav_options if opt["id"] == selected_view), ""
    )
    st.caption(selected_description)

    # Update session state if view changed
    if selected_view != current_view:
        set_session_value("current_view", selected_view)

    # Add experiment selector
    st.divider()
    st.sidebar.header("Experiment Settings")

    client_lab = st.text_input(
        "Client/Lab Name",
        value=get_session_value("client_lab", "NuraxsDemo"),
        key="client_lab_input",
    )

    experiment = st.text_input(
        "Experiment Name",
        value=get_session_value("experiment", "HPLC-Optimization"),
        key="experiment_input",
    )

    output_dir = st.text_input(
        "Output Directory",
        value=get_session_value("output_dir", "hplc_optimization"),
        key="output_dir_input",
    )

    # Update session state with new values
    set_session_value("client_lab", client_lab)
    set_session_value("experiment", experiment)
    set_session_value("output_dir", output_dir)

    return selected_view
