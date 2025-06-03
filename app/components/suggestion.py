"""
Suggestion Component

This module provides the suggestion view for the HPLC Method Optimizer Streamlit app.
"""

import json
import os

import pandas as pd
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
    plot_gradient_profile,
    section_header,
    subsection_header,
    success_box,
    warning_box,
)


def render_suggestion_view():
    """
    Render the suggestion view.
    """
    main_header("Parameter Suggestion")

    st.markdown(
        """
    Get suggestions for the next HPLC method parameters based on Bayesian Optimization.
    These suggestions are designed to efficiently explore the parameter space and converge on optimal conditions.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    # Initialize optimizer service
    optimizer_service = OptimizerService(client_lab, experiment, output_dir)

    # Create tabs for different suggestion steps
    tab1, tab2 = st.tabs(["Get Suggestion", "Current Suggestion"])

    with tab1:
        render_get_suggestion(optimizer_service)

    with tab2:
        render_current_suggestion(optimizer_service)


def render_get_suggestion(optimizer_service: OptimizerService):
    """
    Render the get suggestion section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Get New Suggestion")

    # Check if we have any previous trials
    output_dir = optimizer_service.output_dir
    trials_dir = os.path.join(output_dir, "trials")

    if not os.path.exists(trials_dir):
        os.makedirs(trials_dir, exist_ok=True)

    # Count existing trials
    existing_trials = [
        f for f in os.listdir(trials_dir) if f.startswith("trial_") and f.endswith(".json")
    ]
    num_trials = len(existing_trials)

    if num_trials > 0:
        st.write(f"You have {num_trials} previous trial{'s' if num_trials > 1 else ''}.")
    else:
        info_box(
            """
        No previous trials found. This will be your first trial.
        The suggestion will be based on the parameter space exploration strategy.
        """
        )

    # Compound properties form
    with st.expander("Enter Compound Properties (Optional)", expanded=False):
        st.write("Provide information about your compounds to improve suggestions:")

        # Create a form for compound properties
        compound_properties = {}

        # Basic properties
        col1, col2 = st.columns(2)

        with col1:
            compound_properties["logP"] = st.number_input(
                "LogP",
                min_value=-10.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Octanol-water partition coefficient",
            )

        with col2:
            compound_properties["molecular_weight"] = st.number_input(
                "Molecular Weight (Da)",
                min_value=50.0,
                max_value=2000.0,
                value=300.0,
                step=10.0,
                help="Molecular weight in Daltons",
            )

        # Additional properties
        col1, col2 = st.columns(2)

        with col1:
            compound_properties["pKa"] = st.number_input(
                "pKa",
                min_value=0.0,
                max_value=14.0,
                value=7.0,
                step=0.1,
                help="Acid dissociation constant",
            )

        with col2:
            compound_properties["polar_surface_area"] = st.number_input(
                "Polar Surface Area (Å²)",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=5.0,
                help="Topological polar surface area",
            )

        # Use properties checkbox
        use_properties = st.checkbox(
            "Use these properties for suggestion",
            value=False,
            help="If checked, these properties will be used to generate the suggestion",
        )

    # Get suggestion button
    if st.button("Get Suggestion"):
        # Register process
        process_id = register_process(
            "Suggestion", "Generating parameter suggestion for next trial"
        )

        # Update process status
        update_process(process_id, progress=0.1, status="running")

        try:
            # Get suggestion with or without compound properties
            if use_properties and compound_properties:
                result = optimizer_service.suggest(compound_properties=compound_properties)
                st.session_state.pop("compound_properties", None)  # Clear properties after use
            else:
                result = optimizer_service.suggest()

            if result["success"]:
                # Update process status
                complete_process(
                    process_id,
                    result={
                        "message": f"Suggestion generated for Trial #{result.get('trial_id', 'Unknown')}"
                    },
                )

                # Store suggestion in session state
                set_session_value("current_suggestion", result)

                # Show success message
                success_box(
                    f"Suggestion generated for Trial #{result.get('trial_id', 'Unknown')}! View it in the 'Current Suggestion' tab."
                )

                # Switch to current suggestion tab
                st.experimental_set_query_params(tab="current_suggestion")
            else:
                # Update process status
                fail_process(process_id, error=result.get("error", "Unknown error"))

                # Show error message
                error_box(f"Suggestion failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            # Update process status
            fail_process(process_id, error=str(e))

            # Show error message
            error_box(f"Error getting suggestion: {str(e)}")


def render_current_suggestion(optimizer_service: OptimizerService):
    """
    Render the current suggestion section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Current Suggestion")

    # Get current suggestion from session state
    current_suggestion = get_session_value("current_suggestion")

    if not current_suggestion:
        info_box(
            """
        No current suggestion found. Go to the 'Get Suggestion' tab to generate a new suggestion.
        """
        )
        return

    # Check if suggestion was successful
    if not current_suggestion.get("success", False):
        error_box(f"Last suggestion failed: {current_suggestion.get('error', 'Unknown error')}")
        return

    # Get parameters and trial ID
    parameters = current_suggestion.get("parameters", {})
    trial_id = current_suggestion.get("trial_id", "Unknown")

    if not parameters:
        warning_box("Suggestion does not contain parameters.")
        return

    # Display trial ID
    st.write(f"### Trial #{trial_id}")

    # Display parameters
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display gradient profile
        if (
            "gradient" in parameters
            and "flow_rate" in parameters
            and "pH" in parameters
            and "column_temp" in parameters
        ):
            gradient = parameters["gradient"]
            flow_rate = parameters["flow_rate"]
            pH = parameters["pH"]
            column_temp = parameters["column_temp"]

            fig = plot_gradient_profile(gradient, flow_rate, pH, column_temp)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Display parameter values
        st.write("### Method Parameters")

        st.write(f"**Column Temperature:** {parameters.get('column_temp', 'N/A'):.1f}°C")
        st.write(f"**Flow Rate:** {parameters.get('flow_rate', 'N/A'):.2f} mL/min")
        st.write(f"**pH:** {parameters.get('pH', 'N/A'):.1f}")

        # Display gradient table
        st.write("### Gradient Table")

        if "gradient" in parameters:
            gradient_df = pd.DataFrame(parameters["gradient"], columns=["Time (min)", "%B"])
            st.dataframe(gradient_df)

    # Export options
    subsection_header("Export Options")

    col1, col2 = st.columns(2)

    with col1:
        # Export as JSON
        if st.button("Export as JSON"):
            # Convert to JSON
            json_str = json.dumps(parameters, indent=2)

            # Provide download link
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"trial_{trial_id}_params.json",
                mime="application/json",
            )

    with col2:
        # Export as CSV
        if st.button("Export as CSV"):
            # Create gradient dataframe
            if "gradient" in parameters:
                gradient_df = pd.DataFrame(parameters["gradient"], columns=["Time (min)", "%B"])

                # Add other parameters as metadata
                metadata_df = pd.DataFrame(
                    {
                        "Parameter": ["Column Temperature", "Flow Rate", "pH"],
                        "Value": [
                            f"{parameters.get('column_temp', 'N/A'):.1f}°C",
                            f"{parameters.get('flow_rate', 'N/A'):.2f} mL/min",
                            f"{parameters.get('pH', 'N/A'):.1f}",
                        ],
                    }
                )

                # Combine into a single CSV with a blank line between
                csv_content = (
                    metadata_df.to_csv(index=False) + "\n\n" + gradient_df.to_csv(index=False)
                )

                # Provide download link
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name=f"trial_{trial_id}_params.csv",
                    mime="text/csv",
                )

    # Next steps
    subsection_header("Next Steps")

    st.markdown(
        """
    1. Run an HPLC experiment with these parameters
    2. Export the retention time results as a CSV file
    3. Go to the Reporting tab to report the results
    """
    )

    if st.button("Go to Reporting"):
        set_session_value("current_view", "report")
        st.experimental_rerun()
