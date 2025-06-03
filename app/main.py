"""HPLC Method Optimizer Streamlit App

This module serves as the entry point for the HPLC Method Optimizer Streamlit app.
"""

import streamlit as st

# Import components
from app.components.home import render_home_view
from app.components.simulation import render_simulation_view
from app.components.validation import render_validation_view

# Import services
from app.utils.logging_config import configure_logging

# Import utilities
from app.utils.session import get_session_value, init_session_state, set_session_value

# Configure logging
logger = configure_logging(app_name="streamlit_app")
logger.info("Starting HPLC Method Optimizer Streamlit app")


def main():
    """Main entry point for the Streamlit app."""
    logger.info("Initializing main app function")
    # Set page config
    st.set_page_config(
        page_title="HPLC Method Optimizer",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS for consistent dark theme
    st.markdown(
        """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        color: white;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e84b5;
        color: white;
    }
    div[data-testid="stTable"] {
        background-color: #1e1e1e;
        color: white;
    }
    div[data-testid="stTable"] th {
        background-color: #0e84b5;
        color: white;
    }
    div[data-testid="stTable"] td {
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    logger.info("Page config set")

    # Initialize session state
    logger.info("Initializing session state")
    init_session_state()
    logger.info("Session state initialized")

    # Simple navigation sidebar
    with st.sidebar:
        st.title("Navigation")

        # Get current view from session state
        current_view = get_session_value("current_view", "home")
        logger.info(f"Current view from session state: {current_view}")

        # Navigation options
        selected_view = st.radio(
            "Select View",
            options=["home", "validate", "simulate"],
            format_func=lambda x: (
                "Home" if x == "home" else ("Validation" if x == "validate" else "Simulation")
            ),
            index=0 if current_view == "home" else (1 if current_view == "validate" else 2),
        )

        # Store selected view in session state if changed
        if selected_view != current_view:
            logger.info(f"Changing view from {current_view} to {selected_view}")
            set_session_value("current_view", selected_view)

        # Add experiment settings
        st.sidebar.divider()
        st.sidebar.header("Experiment Settings")

        # Client/Lab input
        client_lab = st.sidebar.text_input(
            "Client/Lab Name", value=get_session_value("client_lab", "NuraxsDemo")
        )
        set_session_value("client_lab", client_lab)

        # Experiment input
        experiment = st.sidebar.text_input(
            "Experiment Name", value=get_session_value("experiment", "HPLC-Optimization")
        )
        set_session_value("experiment", experiment)

        # Output directory input
        output_dir = st.sidebar.text_input(
            "Output Directory", value=get_session_value("output_dir", "hplc_optimization")
        )
        set_session_value("output_dir", output_dir)

    # Display global process status indicator
    active_processes = get_session_value("processes", [])
    active_processes = [p for p in active_processes if p.get("status") == "running"]

    if active_processes:
        st.sidebar.divider()
        st.sidebar.subheader("Running Processes")

        for process in active_processes:
            process_id = process.get("id")
            name = process.get("name")
            description = process.get("description")
            progress = process.get("progress", 0.0)
            # status and start_time are available in the process dict if needed later

            with st.sidebar.expander(f"{name}: {progress:.0%}", expanded=True):
                st.progress(progress)
                st.caption(description)
                if st.button(f"Go to {name}", key=f"goto_{process_id}"):
                    # Determine which view to switch to based on the process name
                    if "validation" in name.lower():
                        set_session_value("current_view", "validate")
                    elif "simulation" in name.lower():
                        set_session_value("current_view", "simulate")
                    st.rerun()

    # Render the selected view
    current_view = get_session_value("current_view", "home")
    logger.info(f"Rendering view: {current_view}")

    try:
        # Get values needed for views
        client_lab = get_session_value("client_lab")
        experiment = get_session_value("experiment")
        output_dir = get_session_value("output_dir")
        # Note: Each view will create its own OptimizerService instance as needed

        if current_view == "home":
            logger.info("Rendering home view")
            render_home_view()
        elif current_view == "validate":
            logger.info("Rendering validation view")
            render_validation_view()
        elif current_view == "simulate":
            logger.info("Rendering simulation view")
            render_simulation_view()
        else:
            logger.error(f"Unknown view: {current_view}")
            st.error(f"Unknown view: {current_view}")
    except Exception as e:
        logger.exception(f"Error rendering view {current_view}: {str(e)}")
        st.error(f"Error rendering view: {str(e)}")
        st.code(f"Error details:\n{str(e)}", language="python")


if __name__ == "__main__":
    main()
