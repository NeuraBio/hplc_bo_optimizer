"""Validation Component

This module provides the validation view for the HPLC Method Optimizer Streamlit app.
"""

import logging
import os
import threading
import time

import streamlit as st

from app.services.optimizer import OptimizerService
from app.utils.session import (
    get_active_processes,
    get_process,
    get_session_value,
    set_session_value,
)

# Get logger
logger = logging.getLogger(__name__)
logger.info("Validation module initialized")


def render_validation_view():
    """
    Render the validation view.
    """
    logger.info("Starting render_validation_view()")
    st.header("Validation")

    # Description
    st.write(
        "Upload historical HPLC data to validate the optimization approach. "
        "This step processes PDF reports to extract retention times, gradients, and other "
        "parameters for use in the optimization process."
    )

    # Check for any active validation processes and display their status

    active_processes = get_active_processes()
    validation_processes = [p for p in active_processes if p.get("name") == "Validation"]

    if validation_processes:
        # Display the most recent validation process
        process = validation_processes[0]
        process_id = process.get("id")

        if process_id:
            # Get the latest process info
            process_info = get_process(process_id)
            if process_info:
                # Create persistent progress containers
                progress = process_info.get("progress", 0.0)
                status_text = st.empty()
                # Use the progress_bar variable to display progress
                st.progress(progress)
                progress_container = st.empty()

                # Display detailed progress info
                if process_info.get("result"):
                    result = process_info["result"]
                    if "message" in result:
                        status_text.text(result["message"])
                    if "processed" in result and "total" in result:
                        processed = result["processed"]
                        total = result["total"]
                        percent = int(progress * 100)
                        progress_detail = f"Processing PDF {processed} of {total} ({percent}%)"
                        progress_container.text(progress_detail)
                else:
                    status_text.text(
                        f"Validation in progress: {process_info.get('description', '')}"
                    )
                    progress_container.text(f"Progress: {int(progress * 100)}%")

                # Show detailed message if available
                if process_info.get("result") and "message" in process_info["result"]:
                    st.text(process_info["result"]["message"])

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
    Render the validation upload form.

    Args:
        optimizer_service: Optimizer service instance
    """
    import os
    import uuid

    from app.utils.session import (
        complete_process,
        fail_process,
        get_process,
        register_process,
        update_process,
    )

    logger.info("Starting render_validation_upload()")
    st.header("Upload PDF Reports")

    # Get output directory from optimizer service
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

    logger.info(f"PDF directory set to: {pdf_dir}")

    # Initialize UI elements that will be updated
    status_text = st.empty()
    progress_bar = st.progress(0)
    progress_detail = st.empty()

    # Function to update progress display
    def update_progress_display(process_id: str) -> bool:
        """Update the progress display and return True if still running."""
        process_info = get_process(process_id)
        if not process_info:
            return False

        status = process_info.get("status")
        progress = process_info.get("progress", 0.0)
        result = process_info.get("result", {})

        # Update progress bar
        progress_bar.progress(progress)

        # Update status text
        if "message" in result:
            status_text.text(result["message"])

        # Update progress details
        if "processed" in result and "total" in result:
            processed = result["processed"]
            total = result["total"]
            percent = int(progress * 100)
            progress_detail.text(f"Processing PDF {processed} of {total} ({percent}%)")

        return status == "running"

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
    pdf_files = [
        os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        st.warning(f"No PDF files found in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files in '{pdf_dir}'")
    st.write(f"Found {len(pdf_files)} PDF files in '{pdf_dir}':")

    # Show list of PDF files
    with st.expander("View PDF Files", expanded=False):
        for pdf_file in pdf_files:
            st.write(f"- {os.path.basename(pdf_file)}")

    # Create a container for the progress UI
    progress_container = st.container()

    # Run validation button
    if st.button("Run Validation"):
        logger.info("Run Validation button clicked")
        logger.info(f"Starting validation of {len(pdf_files)} PDFs from {pdf_dir}")

        # Generate a unique process ID
        process_id = str(uuid.uuid4())

        # Register a new process for validation
        process_id = register_process(
            name="Validation", description=f"Validating {len(pdf_files)} PDF files from {pdf_dir}"
        )
        logger.info(f"Registered validation process with ID: {process_id}")

        # Add to active process tracker to ensure visibility across tabs
        if "active_process_tracker" not in st.session_state:
            st.session_state.active_process_tracker = {}

        # Register this process with a link back to the validation view
        st.session_state.active_process_tracker[process_id] = {
            "view": "validate",
            "name": "Validation",
            "description": f"Validating {len(pdf_files)} PDF files",
        }
        logger.info(f"Added process {process_id} to active process tracker")
        update_process(
            process_id=process_id,
            progress=0.1,
            status="running",
            result={"message": "Starting validation...", "processed": 0, "total": len(pdf_files)},
        )

        # Create progress UI elements with more descriptive initial values
        status_text = progress_container.empty()
        status_text.text("Starting validation...")
        progress_bar = progress_container.progress(0.1)
        progress_detail = progress_container.empty()
        progress_detail.text(f"Processing PDF 0 of {len(pdf_files)} (10%)")

        # Add a placeholder for real-time log output
        log_expander = progress_container.expander("View Processing Log", expanded=False)
        log_output = log_expander.empty()
        log_output.text("Initializing validation...")

        # Create a placeholder for debug info
        debug_expander = progress_container.expander("Debug Info", expanded=False)
        debug_output = debug_expander.empty()

        # Function to update progress display
        def update_progress_display():
            """Update the progress display and return process info."""
            try:
                # Get the latest process info
                process_info = get_process(process_id)
                if not process_info:
                    debug_output.text(f"Process {process_id} not found in session state")
                    return None

                # Extract process information
                status = process_info.get("status")
                progress = process_info.get("progress", 0.0)
                result = process_info.get("result", {})

                # Update debug information
                debug_info = f"Process ID: {process_id}\nStatus: {status}\nProgress: {progress:.2f}\nResult: {result}"
                debug_output.text(debug_info)

                # Update progress bar - force to float
                try:
                    progress_bar.progress(float(progress))
                except Exception as e:
                    logger.error(f"Error updating progress bar: {e}")

                # Update status text
                if "message" in result:
                    status_text.text(result["message"])

                # Update progress details
                if "processed" in result and "total" in result:
                    processed = result["processed"]
                    total = result["total"]
                    percent = int(float(progress) * 100)
                    progress_detail.text(f"Processing PDF {processed} of {total} ({percent}%)")

                # Update log output
                if "output" in result and result["output"]:
                    log_output.text(result["output"])
                elif "message" in result:
                    # Add the message to the log output
                    try:
                        current_log = log_output.text or ""
                        if (
                            result["message"]
                            and current_log
                            and result["message"] not in current_log
                        ):
                            log_output.text(f"{current_log}\n{result['message']}")
                    except Exception as e:
                        logger.error(f"Error updating log: {e}")
                        log_output.text(result["message"])

                return process_info
            except Exception as e:
                logger.error(f"Error in update_progress_display: {e}")
                debug_output.text(f"Error: {str(e)}")
                return None

        # Start the validation in a background thread
        def run_validation():
            try:
                logger.info(f"Starting validation process {process_id}")
                result = optimizer_service.validate(pdf_dir, process_id=process_id)
                if result.get("success"):
                    logger.info(f"Validation completed successfully for {process_id}")
                    complete_process(process_id, result=result)
                else:
                    error = result.get("error", "Validation failed")
                    logger.error(f"Validation failed for {process_id}: {error}")
                    fail_process(process_id, error=error)
            except Exception as e:
                error = str(e)
                logger.exception(f"Error in validation process {process_id}: {error}")
                fail_process(process_id, error=error)

        # Start the validation thread
        validation_thread = threading.Thread(target=run_validation)
        validation_thread.daemon = True
        validation_thread.start()

        # Start a separate thread to update the progress display
        def progress_monitor_thread():
            try:
                # Counter for periodic reruns
                counter = 0

                # Loop until the process is complete or failed
                while True:
                    # Update the progress display
                    process_info = update_progress_display()

                    # Break if the process is done or has an error
                    if not process_info or process_info.get("status") in ["done", "error"]:
                        # One final update
                        update_progress_display()
                        break

                    # Force a rerun every few iterations to ensure UI updates
                    counter += 1
                    if counter >= 5:
                        counter = 0
                        # This is critical for ensuring the UI updates
                        try:
                            st.experimental_rerun()
                        except Exception as e:
                            logger.error(f"Error forcing rerun: {e}")

                    # Sleep briefly to avoid excessive updates
                    time.sleep(0.5)
            except Exception as e:
                logger.exception(f"Error in progress monitor thread: {e}")

        # Start the progress monitor thread
        progress_thread = threading.Thread(target=progress_monitor_thread)
        progress_thread.daemon = True
        progress_thread.start()

        # Wait for the validation thread to complete
        validation_thread.join()

        # Wait for the progress thread to complete (with timeout)
        if "progress_thread" in locals() and progress_thread.is_alive():
            progress_thread.join(timeout=5.0)
            if progress_thread.is_alive():
                logger.warning("Progress monitor thread did not complete in time")

        # Final update to ensure UI reflects the final state
        try:
            final_process_info = update_progress_display()
            # Force a final progress update based on the final status
            if final_process_info and final_process_info.get("status") == "done":
                progress_bar.progress(1.0)
                status_text.text("Validation completed successfully!")
            elif final_process_info and final_process_info.get("status") == "error":
                status_text.text(
                    f"Validation failed: {final_process_info.get('error', 'Unknown error')}"
                )
        except Exception as e:
            logger.error(f"Error in final progress update: {e}")
            final_process_info = get_process(process_id)
            if final_process_info:
                if final_process_info.get("status") == "done":
                    st.success("Validation completed successfully!")

                    # Show summary of results if available
                    result = final_process_info.get("result", {})
                    if "data" in result and isinstance(result["data"], dict):
                        with st.expander("View Validation Summary", expanded=True):
                            st.json(result["data"], expanded=False)

                    # Add a button to view detailed results
                    if st.button("View Detailed Results"):
                        st.session_state["active_tab"] = "Validation Results"
                        st.rerun()

                elif final_process_info.get("status") == "failed":
                    error_msg = final_process_info.get("error", "Unknown error occurred")
                    st.error(f"Validation failed: {error_msg}")

                    # Show error details if available
                    if isinstance(error_msg, dict) and "error" in error_msg:
                        with st.expander("Error Details"):
                            st.error(error_msg["error"])

                    # Add a retry button
                    if st.button("Retry Validation"):
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
