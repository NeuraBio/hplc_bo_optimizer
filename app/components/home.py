"""Home Component

This module provides the home/dashboard view for the HPLC Method Optimizer Streamlit app.
"""

import os

import streamlit as st

from app.utils.session import get_session_value


def render_home_view():
    """
    Render the home/dashboard view.
    """
    st.title("HPLC Method Optimizer")

    st.markdown(
        """
    Welcome to the HPLC Method Optimizer! This application helps you optimize HPLC 
    methods using Bayesian Optimization techniques. Follow the workflow below to get started.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    # Display current experiment info
    st.subheader("Current Experiment")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"**Client/Lab:** {client_lab}")
    with cols[1]:
        st.markdown(f"**Experiment:** {experiment}")
    with cols[2]:
        st.markdown(f"**Output Directory:** {output_dir}")

    # Display workflow steps
    st.subheader("Workflow")

    st.info("Use the sidebar navigation to access different features of the application.")

    # Create workflow cards
    workflow_cols = st.columns(2)

    with workflow_cols[0]:
        st.markdown("### 📊 1. Validation")
        st.markdown("Upload historical data to validate the optimization approach.")
        if st.button("Go to Validation", key="goto_validate"):
            st.session_state.current_view = "validate"
            st.rerun()

    # Display quick tips
    st.subheader("Quick Tips")

    st.info(
        """
    <ul>
        <li><strong>Start with validation</strong> by uploading historical PDF data to validate the scoring function.</li>
        <li><strong>Process PDF reports</strong> to extract retention times, peak widths, and other chromatography data.</li>
    </ul>
    """
    )


def display_experiment_status(output_dir: str):
    """
    Display experiment status if available.

    Args:
        output_dir: Output directory
    """
    # Check if validation directory exists
    validation_dir = os.path.join(output_dir, "validation")

    if not os.path.exists(validation_dir):
        st.info("No validation data available yet. Start by uploading PDF files.")
        return

    st.subheader("Validation Status")

    # List PDF files in validation directory
    pdf_files = [f for f in os.listdir(validation_dir) if f.lower().endswith(".pdf")]

    if pdf_files:
        st.success(f"Found {len(pdf_files)} processed PDF files in validation directory.")
    else:
        st.warning("No processed PDF files found in validation directory.")
