"""
Reporting Component

This module provides the reporting view for the HPLC Method Optimizer Streamlit app.
"""

import io
import json
import os

import pandas as pd
import plotly.express as px
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


def render_reporting_view():
    """
    Render the reporting view.
    """
    main_header("Report Experimental Results")

    st.markdown(
        """
    Report the results of your HPLC experiment to feed back into the Bayesian Optimization process.
    Upload your chromatogram data and provide additional information about the run.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    # Initialize optimizer service
    optimizer_service = OptimizerService(client_lab, experiment, output_dir)

    # Create tabs for different reporting steps
    tab1, tab2, tab3 = st.tabs(["Upload Results", "Review & Submit", "Previous Reports"])

    with tab1:
        render_upload_results(optimizer_service)

    with tab2:
        render_review_submit(optimizer_service)

    with tab3:
        render_previous_reports(optimizer_service)


def render_upload_results(optimizer_service: OptimizerService):
    """
    Render the upload results section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Upload Experimental Results")

    # Check if we have a current suggestion
    current_suggestion = get_session_value("current_suggestion")

    if not current_suggestion or not current_suggestion.get("success", False):
        warning_box(
            """
        No current suggestion found. Go to the Suggestion tab to generate a new suggestion first.
        """
        )

        if st.button("Go to Suggestion"):
            set_session_value("current_view", "suggest")
            st.experimental_rerun()

        return

    # Get trial ID
    trial_id = current_suggestion.get("trial_id", "Unknown")

    st.write(f"Reporting results for Trial #{trial_id}")

    # Upload chromatogram CSV
    st.write("### Upload Chromatogram CSV")

    uploaded_file = st.file_uploader(
        "Upload chromatogram CSV file",
        type=["csv"],
        help="CSV file containing retention time data from your HPLC experiment",
    )

    if uploaded_file is not None:
        try:
            # Read CSV
            chromatogram_df = pd.read_csv(uploaded_file)

            # Store in session state
            set_session_value("current_chromatogram", chromatogram_df.to_dict())

            # Display preview
            st.write("#### Chromatogram Preview")
            st.dataframe(chromatogram_df.head(10))

            # Plot chromatogram
            if "Time" in chromatogram_df.columns and "Signal" in chromatogram_df.columns:
                fig = px.line(chromatogram_df, x="Time", y="Signal", title="Chromatogram")

                st.plotly_chart(fig, use_container_width=True)

            # Success message
            success_box(
                "Chromatogram uploaded successfully! Proceed to 'Review & Submit' to report the results."
            )
        except Exception as e:
            error_box(f"Error reading CSV file: {str(e)}")

    # Manual entry option
    with st.expander("Or Enter Peak Data Manually", expanded=False):
        st.write("If you don't have a CSV file, you can enter peak data manually:")

        # Create empty dataframe if not exists
        if "manual_peaks" not in st.session_state:
            st.session_state.manual_peaks = pd.DataFrame(
                {
                    "Peak": [1],
                    "RT": [0.0],
                    "Height": [0.0],
                    "Area": [0.0],
                    "Width": [0.0],
                    "Tailing": [0.0],
                    "Plates": [0.0],
                }
            )

        # Edit dataframe
        edited_df = st.data_editor(
            st.session_state.manual_peaks, num_rows="dynamic", use_container_width=True
        )

        # Update session state
        st.session_state.manual_peaks = edited_df

        # Store in session state
        if st.button("Save Manual Data"):
            set_session_value("current_chromatogram", {"manual": edited_df.to_dict()})
            success_box(
                "Manual peak data saved! Proceed to 'Review & Submit' to report the results."
            )


def render_review_submit(optimizer_service: OptimizerService):
    """
    Render the review and submit section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Review & Submit Results")

    # Check if we have a current suggestion and chromatogram
    current_suggestion = get_session_value("current_suggestion")
    current_chromatogram = get_session_value("current_chromatogram")

    if not current_suggestion or not current_suggestion.get("success", False):
        info_box(
            """
        No current suggestion found. Go to the Suggestion tab to generate a new suggestion first.
        """
        )
        return

    if not current_chromatogram:
        info_box(
            """
        No chromatogram data found. Go to the 'Upload Results' tab to upload your experimental data.
        """
        )
        return

    # Get trial ID and parameters
    trial_id = current_suggestion.get("trial_id", "Unknown")
    parameters = current_suggestion.get("parameters", {})

    st.write(f"### Trial #{trial_id} - Review")

    # Display parameters
    st.write("#### Method Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Column Temperature", f"{parameters.get('column_temp', 'N/A'):.1f}°C")

    with col2:
        st.metric("Flow Rate", f"{parameters.get('flow_rate', 'N/A'):.2f} mL/min")

    with col3:
        st.metric("pH", f"{parameters.get('pH', 'N/A'):.1f}")

    # Display chromatogram
    st.write("#### Chromatogram Data")

    if "manual" in current_chromatogram:
        # Display manual peak data
        manual_df = pd.DataFrame(current_chromatogram["manual"])
        st.dataframe(manual_df)
    else:
        # Display uploaded chromatogram
        chromatogram_df = pd.DataFrame(current_chromatogram)

        if "Time" in chromatogram_df.columns and "Signal" in chromatogram_df.columns:
            fig = px.line(chromatogram_df, x="Time", y="Signal", title="Chromatogram")

            st.plotly_chart(fig, use_container_width=True)

    # Additional information
    st.write("#### Additional Information")

    col1, col2 = st.columns(2)

    with col1:
        run_quality = st.selectbox(
            "Run Quality",
            options=["Good", "Acceptable", "Poor", "Failed"],
            index=0,
            help="Subjective assessment of the run quality",
        )

    with col2:
        chemist_score = st.slider(
            "Chemist Score (0-10)",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Your subjective score for this run (0 = worst, 10 = best)",
        )

    notes = st.text_area(
        "Notes", value="", help="Any additional notes or observations about this run"
    )

    # Submit button
    if st.button("Submit Results"):
        # Register process
        process_id = register_process("Report", f"Submitting results for Trial #{trial_id}")

        # Update process status
        update_process(process_id, progress=0.1, status="running")

        try:
            # Prepare report data
            report_data = {
                "trial_id": trial_id,
                "parameters": parameters,
                "run_quality": run_quality,
                "chemist_score": chemist_score,
                "notes": notes,
            }

            # Add chromatogram data
            if "manual" in current_chromatogram:
                report_data["chromatogram"] = {"manual": current_chromatogram["manual"]}
            else:
                # Convert DataFrame to CSV string
                chromatogram_df = pd.DataFrame(current_chromatogram)
                csv_buffer = io.StringIO()
                chromatogram_df.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()

                # Add to report data
                report_data["chromatogram"] = {"csv": csv_str}

            # Submit report
            result = optimizer_service.report(trial_id=trial_id, report_data=report_data)

            if result["success"]:
                # Update process status
                complete_process(
                    process_id,
                    result={"message": f"Results for Trial #{trial_id} submitted successfully!"},
                )

                # Clear current chromatogram
                set_session_value("current_chromatogram", None)

                # Show success message
                success_box(f"Results for Trial #{trial_id} submitted successfully!")

                # Offer to get next suggestion
                if st.button("Get Next Suggestion"):
                    set_session_value("current_view", "suggest")
                    st.experimental_rerun()
            else:
                # Update process status
                fail_process(process_id, error=result.get("error", "Unknown error"))

                # Show error message
                error_box(f"Report submission failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            # Update process status
            fail_process(process_id, error=str(e))

            # Show error message
            error_box(f"Error submitting report: {str(e)}")


def render_previous_reports(optimizer_service: OptimizerService):
    """
    Render the previous reports section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Previous Reports")

    # Get output directory
    output_dir = optimizer_service.output_dir
    trials_dir = os.path.join(output_dir, "trials")

    if not os.path.exists(trials_dir):
        info_box("No previous trials found.")
        return

    # Get trial reports
    trial_reports = []

    for filename in os.listdir(trials_dir):
        if filename.startswith("trial_") and filename.endswith(".json"):
            try:
                with open(os.path.join(trials_dir, filename), "r") as f:
                    trial_data = json.load(f)

                    # Extract trial ID from filename
                    trial_id = filename.replace("trial_", "").replace(".json", "")

                    # Add to list
                    trial_reports.append({"trial_id": trial_id, "data": trial_data})
            except Exception as e:
                st.error(f"Error loading trial {filename}: {str(e)}")

    if not trial_reports:
        info_box("No previous trial reports found.")
        return

    # Sort by trial ID
    trial_reports.sort(
        key=lambda x: int(x["trial_id"]) if x["trial_id"].isdigit() else float("inf")
    )

    # Display trial reports
    for trial in trial_reports:
        trial_id = trial["trial_id"]
        trial_data = trial["data"]

        with st.expander(f"Trial #{trial_id}", expanded=False):
            # Display parameters
            st.write("#### Method Parameters")

            parameters = trial_data.get("parameters", {})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Column Temperature", f"{parameters.get('column_temp', 'N/A'):.1f}°C")

            with col2:
                st.metric("Flow Rate", f"{parameters.get('flow_rate', 'N/A'):.2f} mL/min")

            with col3:
                st.metric("pH", f"{parameters.get('pH', 'N/A'):.1f}")

            # Display score if available
            if "score" in trial_data:
                st.metric("Score", f"{trial_data['score']:.2f}")

            # Display chromatogram if available
            if "chromatogram_path" in trial_data:
                chromatogram_path = trial_data["chromatogram_path"]

                if os.path.exists(chromatogram_path):
                    try:
                        chromatogram_df = pd.read_csv(chromatogram_path)

                        if (
                            "Time" in chromatogram_df.columns
                            and "Signal" in chromatogram_df.columns
                        ):
                            fig = px.line(
                                chromatogram_df, x="Time", y="Signal", title="Chromatogram"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading chromatogram: {str(e)}")

            # Display additional information
            if "run_quality" in trial_data:
                st.write(f"**Run Quality:** {trial_data['run_quality']}")

            if "chemist_score" in trial_data:
                st.write(f"**Chemist Score:** {trial_data['chemist_score']}/10")

            if "notes" in trial_data and trial_data["notes"]:
                st.write(f"**Notes:** {trial_data['notes']}")

    # Provide download link for all trials
    if st.button("Download All Trial Reports"):
        # Create a ZIP file with all trial reports
        try:
            import zipfile
            from io import BytesIO

            # Create ZIP file in memory
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename in os.listdir(trials_dir):
                    if filename.startswith("trial_") and filename.endswith(".json"):
                        file_path = os.path.join(trials_dir, filename)
                        zip_file.write(file_path, filename)

            # Reset buffer position
            zip_buffer.seek(0)

            # Provide download link
            st.download_button(
                label="Download ZIP",
                data=zip_buffer,
                file_name="trial_reports.zip",
                mime="application/zip",
            )
        except Exception as e:
            error_box(f"Error creating ZIP file: {str(e)}")
