"""Validation Component

This module provides the validation view for the HPLC Method Optimizer Streamlit app.
"""

import logging
import os
import time

import streamlit as st

from app.services.optimizer import OptimizerService
from app.utils.session import get_session_value, set_session_value

# Get logger
logger = logging.getLogger(__name__)
logger.info("Validation module initialized")


def render_validation_view():
    """
    Render the validation view.
    """
    logger.info("Rendering validation view")
    st.title("Validation")

    st.markdown(
        """
    Upload historical HPLC data to validate the optimization approach. This step processes PDF reports
    to extract retention times, gradients, and other parameters for use in the optimization process.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    # Initialize optimizer service
    logger.info("Initializing OptimizerService")
    optimizer_service = OptimizerService(client_lab, experiment, output_dir)

    # Create tabs for different validation steps
    tab1, tab2 = st.tabs(["Upload & Validate", "Validation Results"])

    with tab1:
        logger.info("Rendering validation upload tab")
        render_validation_upload(optimizer_service)

    with tab2:
        logger.info("Rendering validation results tab")
        render_validation_results(optimizer_service)

    logger.info("Completed rendering validation view")


def render_validation_upload(optimizer_service: OptimizerService):
    """
    Render the validation upload view.

    Args:
        optimizer_service: Optimizer service instance
    """
    logger.info("Starting render_validation_upload()")
    st.header("Upload PDF Reports")

    # Get output directory
    output_dir = optimizer_service.output_dir

    # Create validation directory if it doesn't exist
    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    # Default PDF directory
    default_pdf_dir = os.path.join("data", "pdfs")

    # PDF directory input
    pdf_dir = st.text_input(
        "PDF Directory",
        value=get_session_value("pdf_dir", default_pdf_dir),
        help="Directory containing PDF reports to process",
    )

    # Update session state
    set_session_value("pdf_dir", pdf_dir)

    # Get output directory from session state
    output_dir = get_session_value("output_dir", "/app/output")
    logger.info(f"Output directory: {output_dir}")

    # Check if output directory exists
    if not os.path.exists(output_dir):
        logger.warning(f"Output directory not found: {output_dir}")
        st.warning(f"Output directory not found: {output_dir}")
        return

    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        logger.warning(f"PDF directory not found: {pdf_dir}")
        st.warning(f"Directory '{pdf_dir}' does not exist.")

        # Add option to create directory
        if st.button("Create Directory"):
            try:
                os.makedirs(pdf_dir, exist_ok=True)
                st.success(f"Directory '{pdf_dir}' created successfully.")
                st.rerun()  # Refresh to show the directory contents
            except Exception as e:
                logger.error(f"Error creating directory: {str(e)}")
                st.error(f"Error creating directory: {str(e)}")
        return

    # Check if PDF directory contains PDF files
    logger.info(f"Checking PDF directory: {pdf_dir}")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        st.warning(f"No PDF files found in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files in '{pdf_dir}'")
    st.write(f"Found {len(pdf_files)} PDF files in '{pdf_dir}':")

    # Show list of PDF files
    with st.expander("View PDF Files", expanded=False):
        for pdf_file in pdf_files:
            st.write(f"- {pdf_file}")

    # Run validation button
    if st.button("Run Validation"):
        logger.info("Run Validation button clicked")
        logger.info("Starting validation process")
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting validation...")
        logger.info(f"Starting validation with PDF directory: {pdf_dir}")

        try:
            # Create a simulated validation process with progress updates
            # This is a fallback if the actual validation doesn't show progress
            for i in range(10):
                # Update progress
                progress = (i + 1) / 10
                progress_bar.progress(progress)
                status_message = f"Processing PDF files... {(i + 1) * 10}%"
                logger.info(status_message)
                status_text.text(status_message)

                # Run actual validation on the last step
                if i == 9:
                    logger.info(f"Running validation with optimizer_service.validate({pdf_dir})")
                    # Run the actual validation through the optimizer service
                    try:
                        logger.info("Calling optimizer_service.validate()")
                        result = optimizer_service.validate(pdf_dir)
                        logger.info(f"Validation result: {result}")

                        if result["success"]:
                            logger.info("Validation completed successfully")
                            status_text.text("Validation completed successfully!")
                            st.success(
                                "Validation completed successfully! View results in the 'Validation Results' tab."
                            )
                        else:
                            logger.error(
                                f"Validation failed: {result.get('error', 'Unknown error')}"
                            )
                            status_text.text("Validation failed.")
                            st.error(f"Validation failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.exception(f"Exception during optimizer_service.validate: {str(e)}")
                        status_text.text(f"Error during validation: {str(e)}")
                        st.error(f"Error during validation: {str(e)}")

                # Simulate processing time
                if i < 9:  # Don't sleep on the last iteration
                    time.sleep(0.3)

        except Exception as e:
            # Show error message
            error_message = f"Error running validation: {str(e)}"
            logger.exception(error_message)
            status_text.text("Validation failed with an error.")
            st.error(error_message)

        # Add a refresh button to see results
        if st.button("View Results"):
            logger.info("User clicked View Results button, rerunning app")
            st.rerun()


def render_validation_results(optimizer_service: OptimizerService):
    """
    Render the validation results view.

    Args:
        optimizer_service: Optimizer service instance
    """
    logger.info("Starting render_validation_results()")
    st.header("Validation Results")

    # Get validation directory
    validation_dir = os.path.join(optimizer_service.output_dir, "validation")

    # Check if validation directory exists
    if not os.path.exists(validation_dir):
        st.info("No validation results available. Run validation first.")
        return

    try:
        # Load validation details
        import json

        with open(os.path.join(validation_dir, "validation_details.json"), "r") as f:
            validation_data = json.load(f)

        # Display validation summary
        if "compounds" in validation_data:
            st.success(f"Successfully validated {len(validation_data['compounds'])} compounds")

            # Display compounds in an expander
            with st.expander("View Validated Compounds", expanded=False):
                for compound in validation_data["compounds"]:
                    st.write(f"- {compound}")

        # Display validation report if available
        if os.path.exists(os.path.join(validation_dir, "validation_report.html")):
            st.subheader("Validation Report")
            st.markdown(
                f"""<a href="file://{os.path.join(validation_dir, 'validation_report.html')}" target="_blank">Open Validation Report</a>""",
                unsafe_allow_html=True,
            )

            # Display link to download report
            with open(os.path.join(validation_dir, "validation_report.html"), "rb") as file:
                st.download_button(
                    label="Download Validation Report",
                    data=file,
                    file_name="validation_report.html",
                    mime="text/html",
                )

        # Display validation summary if available
        if os.path.exists(os.path.join(validation_dir, "validation_summary.csv")):
            st.subheader("Validation Summary")

            # Load validation summary
            import pandas as pd

            try:
                summary_df = pd.read_csv(os.path.join(validation_dir, "validation_summary.csv"))
                st.dataframe(summary_df)
            except Exception as e:
                st.error(f"Error loading validation summary: {str(e)}")

        # Display plots if available
        plots_dir = os.path.join(validation_dir, "plots")
        if os.path.exists(plots_dir):
            st.subheader("Validation Plots")

            plot_files = [
                f for f in os.listdir(plots_dir) if f.endswith(".png") or f.endswith(".jpg")
            ]

            if plot_files:
                for plot_file in plot_files:
                    plot_path = os.path.join(plots_dir, plot_file)
                    st.image(plot_path, caption=plot_file)
            else:
                st.info("No validation plots available.")

    except Exception as e:
        st.error(f"Error loading validation results: {str(e)}")

    # End of render_validation_results function
