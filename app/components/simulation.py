"""
Simulation Component

This module provides the simulation view for the HPLC Method Optimizer Streamlit app.
"""

import logging
import os
import time

import streamlit as st

from app.services.optimizer import OptimizerService
from app.utils.session import (
    complete_process,
    fail_process,
    get_session_value,
    register_process,
    set_session_value,
    update_process,
)
from app.utils.ui_helpers import (
    error_box,
    info_box,
    main_header,
    section_header,
    success_box,
    warning_box,
)

# Configure logger
logger = logging.getLogger(__name__)


def render_simulation_view():
    """
    Render the simulation view.
    """
    logger.info("Rendering simulation view")
    main_header("Bayesian Optimization Simulation")

    st.markdown(
        """
    Run a simulation to test how Bayesian Optimization would perform using your historical data.
    This helps validate the optimization approach before running actual experiments.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    logger.info(
        f"Using client_lab: {client_lab}, experiment: {experiment}, output_dir: {output_dir}"
    )

    # Initialize optimizer service
    optimizer_service = OptimizerService(client_lab, experiment, output_dir)

    # Create tabs for different simulation steps
    tab1, tab2 = st.tabs(["Configure & Run", "Simulation Results"])

    with tab1:
        logger.info("Rendering simulation configuration tab")
        render_simulation_config(optimizer_service)

    with tab2:
        logger.info("Rendering simulation results tab")
        render_simulation_results(optimizer_service)

    logger.info("Completed rendering simulation view")


def render_simulation_config(optimizer_service: OptimizerService):
    """
    Render the simulation configuration section.

    Args:
        optimizer_service: Optimizer service instance
    """
    logger.info("Starting render_simulation_config()")
    section_header("Configure Simulation")

    # Get output directory
    output_dir = optimizer_service.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Check if validation results exist
    validation_dir = os.path.join(output_dir, "validation")
    validation_file = os.path.join(validation_dir, "validation_details.json")
    logger.info(f"Checking validation file: {validation_file}")

    # Allow custom validation file path input
    with st.expander("Advanced Options", expanded=False):
        custom_validation_file = st.text_input(
            "Custom Validation File",
            value=validation_file,
            help="Path to a custom validation details JSON file. Use this if the default path doesn't work.",
        )
        if custom_validation_file and custom_validation_file != validation_file:
            validation_file = custom_validation_file
            logger.info(f"Using custom validation file: {validation_file}")

    if not os.path.exists(validation_file):
        logger.warning(f"Validation file not found: {validation_file}")
        warning_box(
            """
        No validation results found. Please run validation first to extract historical data from PDF reports.
        Go to the Validation tab to process your historical data.
        """
        )

        if st.button("Go to Validation"):
            logger.info("User clicked 'Go to Validation' button")
            set_session_value("current_view", "validate")
            st.rerun()

        return

    # Simulation parameters
    st.write("Configure the simulation parameters:")

    col1, col2 = st.columns(2)

    with col1:
        n_trials = st.slider(
            "Number of Trials",
            min_value=5,
            max_value=100,
            value=get_session_value("n_trials", 20),
            step=5,
            help="Number of trials to simulate",
        )

        # Update session state
        set_session_value("n_trials", n_trials)

    with col2:
        use_vector_similarity = st.checkbox(
            "Use Vector Similarity",
            value=get_session_value("use_vector_similarity", True),
            help="Use vector similarity for matching runs (recommended)",
        )

        # Update session state
        set_session_value("use_vector_similarity", use_vector_similarity)

        if use_vector_similarity:
            similarity_metric = st.selectbox(
                "Similarity Metric",
                options=["cosine", "euclidean", "correlation", "manhattan"],
                index=["cosine", "euclidean", "correlation", "manhattan"].index(
                    get_session_value("similarity_metric", "cosine")
                ),
                help="Distance metric to use for vector similarity",
            )

            # Update session state
            set_session_value("similarity_metric", similarity_metric)

    # Advanced options expander
    with st.expander("Advanced Options", expanded=True):  # Set to expanded by default
        custom_validation_file = st.text_input(
            "Custom Validation File",
            value=get_session_value(
                "custom_validation_file",
                "/app/hplc_optimization/validation/validation_details.json",
            ),
            help="Path to the validation details JSON file. Inside the Docker container, your project is mounted at /app",
        )

        # Update session state
        set_session_value("custom_validation_file", custom_validation_file)

    # Run simulation button
    if st.button("Run Simulation", key="run_simulation"):
        # Use custom validation file if provided
        if custom_validation_file:
            validation_file = custom_validation_file

        logger.info(
            f"Running simulation with {n_trials} trials, vector similarity: {use_vector_similarity}, metric: {similarity_metric}"
        )
        logger.info(f"Using validation file: {validation_file}")

        # Verify validation file exists
        if not os.path.exists(validation_file):
            error_message = f"Validation file not found: {validation_file}"
            logger.error(error_message)
            error_box(f"Simulation failed: {error_message}")
            return

        # Register process
        process_id = register_process("Simulation", f"Running BO simulation with {n_trials} trials")

        # Update process status
        update_process(process_id, progress=0.1, status="running")
        logger.info(f"Registered simulation process with ID: {process_id}")

        # Create progress bar
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        status_text.text("Running simulation...")

        # Run simulation in a separate thread to avoid blocking the UI
        result = {"success": False, "error": "Simulation failed to start"}

        try:
            # Run simulation with progress updates
            validation_file_path = custom_validation_file if custom_validation_file else None
            logger.info(f"Starting simulation with validation_file_path: {validation_file_path}")

            # Update progress to show we're starting
            progress_bar.progress(0.1)
            status_text.text("Initializing simulation...")

            # Simulate progress updates during the simulation
            # The actual simulation happens in one call, but we can show progress to the user
            for i in range(10):
                # Update progress
                progress = 0.1 + (i + 1) * 0.08  # Goes from 10% to 90%
                progress_bar.progress(progress)
                status_message = f"Running simulation... {int(progress * 100)}%"
                logger.info(status_message)
                status_text.text(status_message)

                # Run actual simulation on the last step
                if i == 9:
                    logger.info("Executing optimizer_service.simulate()")
                    result = optimizer_service.simulate(
                        n_trials=n_trials,
                        validation_file=validation_file_path,
                        use_vector_similarity=use_vector_similarity,
                        similarity_metric=similarity_metric if use_vector_similarity else "cosine",
                    )
                    logger.info(f"Simulation result: {result['success']}")

                # Simulate processing time
                if i < 9:  # Don't sleep on the last iteration
                    time.sleep(0.3)

            # Final progress update
            progress_bar.progress(1.0)

            if result["success"]:
                # Update process status
                complete_process(
                    process_id,
                    result={
                        "message": "Simulation completed successfully!",
                        "report_path": result.get("report_path"),
                    },
                )
                logger.info("Simulation completed successfully")

                # Show success message
                status_text.text("Simulation completed successfully!")
                success_box(
                    "Simulation completed successfully! View results in the 'Simulation Results' tab."
                )
            else:
                # Update process status
                fail_process(process_id, error=result.get("error", "Unknown error"))
                logger.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

                # Show error message
                status_text.text("Simulation failed.")
                error_box(f"Simulation failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            # Update process status
            fail_process(process_id, error=str(e))
            logger.exception(f"Exception during simulation: {str(e)}")

            # Show error message
            status_text.text("Simulation failed with an error.")
            error_box(f"Error running simulation: {str(e)}")

        # Add a refresh button to see results
        if st.button("View Results"):
            logger.info("User clicked View Results button, rerunning app")
            st.rerun()


def render_simulation_results(optimizer_service: OptimizerService):
    """
    Render the simulation results section.

    Args:
        optimizer_service: Optimizer service instance
    """
    logger.info("Starting render_simulation_results()")
    section_header("Simulation Results")

    # Get output directory
    output_dir = optimizer_service.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Check if simulation results exist
    simulation_dir = os.path.join(output_dir, "bo_simulation")
    report_file = os.path.join(simulation_dir, "bo_simulation_report.html")
    logger.info(f"Checking simulation report: {report_file}")

    if not os.path.exists(simulation_dir) or not os.path.exists(report_file):
        logger.info("No simulation results found")
        info_box("No simulation results found. Please run a simulation first.")
        return

    # Display the HTML report in an iframe
    logger.info(f"Displaying simulation report from {report_file}")

    # Read the HTML file
    try:
        with open(report_file, "r") as f:
            html_content = f.read()

        # Display the HTML report
        st.write("### Simulation Report")

        # Check if plot images exist and display them directly
        plots_dir = os.path.join(simulation_dir, "plots")
        bo_performance_plot = os.path.join(plots_dir, "bo_performance.png")
        bo_vs_manual_plot = os.path.join(plots_dir, "bo_vs_manual.png")

        # Display section headers and plots
        st.write("#### Performance Plots")
        if os.path.exists(bo_performance_plot):
            st.image(bo_performance_plot, caption="BO Performance")
        else:
            st.warning("BO Performance plot not found")

        st.write("#### BO vs Manual Progression Comparison")
        if os.path.exists(bo_vs_manual_plot):
            st.image(bo_vs_manual_plot, caption="BO vs Manual Comparison")
            st.write(
                "This plot compares how quickly Bayesian Optimization (red) finds optimal solutions compared to the chronological manual experimentation (green). The dots represent points where the best score was improved."
            )
        else:
            st.warning("BO vs Manual Comparison plot not found")

        # Also display the full HTML report for completeness
        st.write("#### Full HTML Report")
        st.components.v1.html(html_content, height=600, scrolling=True)

        # Add a download button for the report
        with open(report_file, "rb") as file:
            st.download_button(
                label="Download Full Report",
                data=file,
                file_name="bo_simulation_report.html",
                mime="text/html",
            )

        # Add a note about the simulation results
        st.info(
            """
        The simulation report contains detailed information about the Bayesian Optimization simulation, including:
        - Convergence plots showing how the score improves over trials
        - Parameter importance analysis
        - Detailed trial results
        - Contour plots showing the optimization landscape
        
        You can download the full HTML report using the button above for offline viewing.
        """
        )

        # Add a link to run a new simulation
        if st.button("Run New Simulation"):
            # Switch to the configuration tab
            st.session_state.simulation_tab = "config"
            st.rerun()

    except Exception as e:
        logger.exception(f"Error displaying simulation report: {e}")
        error_box(f"Error displaying simulation report: {str(e)}")
        return

    logger.info("Completed rendering simulation results")
