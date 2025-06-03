"""
Status Component

This module provides the status view for the HPLC Method Optimizer Streamlit app.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.services.optimizer import OptimizerService
from app.utils.session import get_session_value
from app.utils.ui_helpers import (
    info_box,
    main_header,
    section_header,
    subsection_header,
    warning_box,
)


def render_status_view():
    """
    Render the status view.
    """
    main_header("Experiment Status")

    st.markdown(
        """
    View the current status of your HPLC method optimization experiment.
    This page provides an overview of all trials, convergence metrics, and optimization progress.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    # Initialize optimizer service
    optimizer_service = OptimizerService(client_lab, experiment, output_dir)

    # Create tabs for different status views
    tab1, tab2, tab3 = st.tabs(["Overview", "Trial Details", "Parameter Trends"])

    with tab1:
        render_overview(optimizer_service)

    with tab2:
        render_trial_details(optimizer_service)

    with tab3:
        render_parameter_trends(optimizer_service)


def render_overview(optimizer_service: OptimizerService):
    """
    Render the overview section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Experiment Overview")

    # Get output directory
    output_dir = optimizer_service.output_dir
    trials_dir = os.path.join(output_dir, "trials")

    if not os.path.exists(trials_dir):
        info_box("No trials found for this experiment.")
        return

    # Get trial data
    trial_data = load_trial_data(trials_dir)

    if not trial_data:
        info_box("No trial data found for this experiment.")
        return

    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trials", len(trial_data))

    with col2:
        # Calculate best score
        scores = [trial.get("score", 0) for trial in trial_data if "score" in trial]
        best_score = max(scores) if scores else 0

        st.metric("Best Score", f"{best_score:.2f}")

    with col3:
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0

        st.metric("Average Score", f"{avg_score:.2f}")

    with col4:
        # Calculate trials to reach 90% of max score
        if scores:
            max_score = max(scores)
            threshold = 0.9 * max_score
            trials_to_threshold = next(
                (i + 1 for i, score in enumerate(scores) if score >= threshold), len(scores)
            )

            st.metric("Trials to 90% Max", trials_to_threshold)

    # Create convergence plot
    subsection_header("Optimization Convergence")

    if scores:
        # Create dataframe for plotting
        df = pd.DataFrame(
            {
                "Trial": list(range(1, len(scores) + 1)),
                "Score": scores,
                "Best Score": np.maximum.accumulate(scores),
            }
        )

        # Create convergence plot
        fig = go.Figure()

        # Add individual trial scores
        fig.add_trace(
            go.Scatter(
                x=df["Trial"],
                y=df["Score"],
                mode="markers",
                name="Trial Score",
                marker=dict(size=8, color="#0066cc"),
            )
        )

        # Add best score line
        fig.add_trace(
            go.Scatter(
                x=df["Trial"],
                y=df["Best Score"],
                mode="lines",
                name="Best Score",
                line=dict(color="#00cc66", width=3),
            )
        )

        # Update layout
        fig.update_layout(
            title="Optimization Convergence",
            xaxis_title="Trial Number",
            yaxis_title="Score",
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Create parameter distribution plot
    subsection_header("Parameter Distributions")

    # Extract parameters
    ph_values = [trial.get("parameters", {}).get("pH", None) for trial in trial_data]
    ph_values = [ph for ph in ph_values if ph is not None]

    temp_values = [trial.get("parameters", {}).get("column_temp", None) for trial in trial_data]
    temp_values = [temp for temp in temp_values if temp is not None]

    flow_values = [trial.get("parameters", {}).get("flow_rate", None) for trial in trial_data]
    flow_values = [flow for flow in flow_values if flow is not None]

    # Create parameter distribution plots
    if ph_values and temp_values and flow_values:
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(x=ph_values, nbins=10, title="pH Distribution", labels={"x": "pH"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                x=temp_values,
                nbins=10,
                title="Temperature Distribution",
                labels={"x": "Temperature (°C)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(
                x=flow_values,
                nbins=10,
                title="Flow Rate Distribution",
                labels={"x": "Flow Rate (mL/min)"},
            )
            st.plotly_chart(fig, use_container_width=True)


def render_trial_details(optimizer_service: OptimizerService):
    """
    Render the trial details section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Trial Details")

    # Get output directory
    output_dir = optimizer_service.output_dir
    trials_dir = os.path.join(output_dir, "trials")

    if not os.path.exists(trials_dir):
        info_box("No trials found for this experiment.")
        return

    # Get trial data
    trial_data = load_trial_data(trials_dir)

    if not trial_data:
        info_box("No trial data found for this experiment.")
        return

    # Create dataframe for display
    rows = []

    for trial in trial_data:
        trial_id = trial.get("trial_id", "Unknown")
        parameters = trial.get("parameters", {})

        row = {
            "Trial ID": trial_id,
            "pH": parameters.get("pH", "N/A"),
            "Temperature (°C)": parameters.get("column_temp", "N/A"),
            "Flow Rate (mL/min)": parameters.get("flow_rate", "N/A"),
            "Score": trial.get("score", "N/A"),
            "Run Quality": trial.get("run_quality", "N/A"),
        }

        rows.append(row)

    # Create dataframe
    df = pd.DataFrame(rows)

    # Sort by Trial ID
    df["Trial ID"] = pd.to_numeric(df["Trial ID"], errors="coerce")
    df = df.sort_values("Trial ID")

    # Display dataframe
    st.dataframe(df, use_container_width=True)

    # Allow download as CSV
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Trial Details CSV",
        csv,
        "trial_details.csv",
        "text/csv",
        key="download-trial-details-csv",
    )

    # Display individual trial details
    subsection_header("Individual Trial Details")

    # Create selectbox for trial selection
    trial_ids = [trial.get("trial_id", "Unknown") for trial in trial_data]
    selected_trial_id = st.selectbox("Select Trial", trial_ids)

    # Get selected trial data
    selected_trial = next(
        (trial for trial in trial_data if trial.get("trial_id", "Unknown") == selected_trial_id),
        None,
    )

    if selected_trial:
        # Display trial details
        st.write(f"### Trial #{selected_trial_id}")

        # Display parameters
        parameters = selected_trial.get("parameters", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("pH", f"{parameters.get('pH', 'N/A')}")

        with col2:
            st.metric("Temperature", f"{parameters.get('column_temp', 'N/A')}°C")

        with col3:
            st.metric("Flow Rate", f"{parameters.get('flow_rate', 'N/A')} mL/min")

        # Display score
        if "score" in selected_trial:
            st.metric("Score", f"{selected_trial['score']:.2f}")

        # Display gradient
        if "parameters" in selected_trial and "gradient" in parameters:
            gradient = parameters["gradient"]

            # Create dataframe
            gradient_df = pd.DataFrame(gradient, columns=["Time (min)", "%B"])

            # Display gradient table
            st.write("#### Gradient Table")
            st.dataframe(gradient_df, use_container_width=True)

            # Plot gradient profile
            fig = px.line(
                gradient_df, x="Time (min)", y="%B", title="Gradient Profile", markers=True
            )

            st.plotly_chart(fig, use_container_width=True)

        # Display chromatogram
        chromatogram_path = selected_trial.get("chromatogram_path")

        if chromatogram_path and os.path.exists(chromatogram_path):
            st.write("#### Chromatogram")

            try:
                chromatogram_df = pd.read_csv(chromatogram_path)

                if "Time" in chromatogram_df.columns and "Signal" in chromatogram_df.columns:
                    fig = px.line(chromatogram_df, x="Time", y="Signal", title="Chromatogram")

                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading chromatogram: {str(e)}")

        # Display additional information
        st.write("#### Additional Information")

        if "run_quality" in selected_trial:
            st.write(f"**Run Quality:** {selected_trial['run_quality']}")

        if "chemist_score" in selected_trial:
            st.write(f"**Chemist Score:** {selected_trial['chemist_score']}/10")

        if "notes" in selected_trial and selected_trial["notes"]:
            st.write(f"**Notes:** {selected_trial['notes']}")


def render_parameter_trends(optimizer_service: OptimizerService):
    """
    Render the parameter trends section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Parameter Trends")

    # Get output directory
    output_dir = optimizer_service.output_dir
    trials_dir = os.path.join(output_dir, "trials")

    if not os.path.exists(trials_dir):
        info_box("No trials found for this experiment.")
        return

    # Get trial data
    trial_data = load_trial_data(trials_dir)

    if not trial_data:
        info_box("No trial data found for this experiment.")
        return

    # Create dataframe for analysis
    rows = []

    for trial in trial_data:
        trial_id = trial.get("trial_id", "Unknown")
        parameters = trial.get("parameters", {})

        row = {
            "Trial ID": trial_id,
            "pH": parameters.get("pH", None),
            "Temperature": parameters.get("column_temp", None),
            "Flow Rate": parameters.get("flow_rate", None),
            "Score": trial.get("score", None),
        }

        rows.append(row)

    # Create dataframe
    df = pd.DataFrame(rows)

    # Convert Trial ID to numeric
    df["Trial ID"] = pd.to_numeric(df["Trial ID"], errors="coerce")

    # Sort by Trial ID
    df = df.sort_values("Trial ID")

    # Drop rows with missing values
    df = df.dropna()

    if df.empty:
        warning_box("Not enough data to analyze parameter trends.")
        return

    # Create parameter trend plots
    subsection_header("Parameter vs. Score")

    # Create parameter vs. score plots
    col1, col2, col3 = st.columns(3)

    with col1:
        if "pH" in df.columns and "Score" in df.columns:
            fig = px.scatter(
                df,
                x="pH",
                y="Score",
                title="pH vs. Score",
                trendline="ols",
                hover_data=["Trial ID"],
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Temperature" in df.columns and "Score" in df.columns:
            fig = px.scatter(
                df,
                x="Temperature",
                y="Score",
                title="Temperature vs. Score",
                trendline="ols",
                hover_data=["Trial ID"],
            )

            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if "Flow Rate" in df.columns and "Score" in df.columns:
            fig = px.scatter(
                df,
                x="Flow Rate",
                y="Score",
                title="Flow Rate vs. Score",
                trendline="ols",
                hover_data=["Trial ID"],
            )

            st.plotly_chart(fig, use_container_width=True)

    # Create parameter trend over time plots
    subsection_header("Parameter Trends Over Time")

    # Create parameter trend over time plots
    col1, col2, col3 = st.columns(3)

    with col1:
        if "pH" in df.columns and "Trial ID" in df.columns:
            fig = px.line(df, x="Trial ID", y="pH", title="pH Over Time", markers=True)

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Temperature" in df.columns and "Trial ID" in df.columns:
            fig = px.line(
                df, x="Trial ID", y="Temperature", title="Temperature Over Time", markers=True
            )

            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if "Flow Rate" in df.columns and "Trial ID" in df.columns:
            fig = px.line(
                df, x="Trial ID", y="Flow Rate", title="Flow Rate Over Time", markers=True
            )

            st.plotly_chart(fig, use_container_width=True)

    # Create score over time plot
    if "Score" in df.columns and "Trial ID" in df.columns:
        fig = px.line(df, x="Trial ID", y="Score", title="Score Over Time", markers=True)

        st.plotly_chart(fig, use_container_width=True)

    # Create parameter correlation heatmap
    subsection_header("Parameter Correlations")

    # Calculate correlations
    corr = df[["pH", "Temperature", "Flow Rate", "Score"]].corr()

    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Parameter Correlation Heatmap",
    )

    st.plotly_chart(fig, use_container_width=True)


def load_trial_data(trials_dir: str) -> List[Dict[str, Any]]:
    """
    Load trial data from the trials directory.

    Args:
        trials_dir: Path to the trials directory

    Returns:
        List of trial data dictionaries
    """
    trial_data = []

    for filename in os.listdir(trials_dir):
        if filename.startswith("trial_") and filename.endswith(".json"):
            try:
                with open(os.path.join(trials_dir, filename), "r") as f:
                    data = json.load(f)

                    # Extract trial ID from filename
                    trial_id = filename.replace("trial_", "").replace(".json", "")

                    # Add trial ID to data
                    data["trial_id"] = trial_id

                    # Add to list
                    trial_data.append(data)
            except Exception as e:
                st.error(f"Error loading trial {filename}: {str(e)}")

    return trial_data
