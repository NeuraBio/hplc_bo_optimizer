"""
UI Helper Functions

This module provides helper functions for building consistent UI components.
"""

import base64
import time
from typing import List, Optional, Union

import plotly.graph_objects as go
import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #555;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9900;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e6fff3;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00cc66;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #cc0000;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .stProgress > div > div > div > div {
        background-color: #0066cc;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main_header(text: str):
    """
    Display a main header with consistent styling.

    Args:
        text: Header text
    """
    st.title(text)


def section_header(text: str):
    """
    Display a section header with consistent styling.

    Args:
        text: Header text
    """
    st.header(text)


def subsection_header(text: str):
    """
    Display a subsection header with consistent styling.

    Args:
        text: Header text
    """
    st.subheader(text)


def info_box(text: str):
    """
    Display an info box with consistent styling.

    Args:
        text: Info text
    """
    st.info(text)


def warning_box(text: str):
    """
    Display a warning box with consistent styling.

    Args:
        text: Warning text
    """
    st.warning(text)


def success_box(text: str):
    """
    Display a success box with consistent styling.

    Args:
        text: Success text
    """
    st.success(text)


def error_box(text: str):
    """
    Display an error box with consistent styling.

    Args:
        text: Error text
    """
    st.error(text)


def card(content_function, key: Optional[str] = None):
    """
    Display content in a card with consistent styling.

    Args:
        content_function: Function that renders the card content
        key: Optional key for the container
    """
    with st.container(key=key):
        content_function()


def metric_card(
    value: Union[int, float, str], label: str, delta: Optional[Union[int, float]] = None
):
    """
    Display a metric card with a value and label.

    Args:
        value: Metric value
        label: Metric label
        delta: Optional delta value for the metric
    """
    # Format delta if provided
    delta_str = f"{delta:+g}" if delta is not None else None

    # Display metric
    st.metric(label=label, value=value, delta=delta_str)


def format_time_elapsed(start_time: float) -> str:
    """
    Format elapsed time in a human-readable format.

    Args:
        start_time: Start time in seconds since epoch

    Returns:
        Formatted time string
    """
    elapsed = time.time() - start_time

    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    elif elapsed < 3600:
        minutes = elapsed / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = elapsed / 3600
        return f"{hours:.1f} hours"


def plot_gradient_profile(
    gradient: List[List[float]], flow_rate: float, pH: float, column_temp: float
) -> go.Figure:
    """
    Create a plotly figure of the gradient profile.

    Args:
        gradient: List of [time, %B] points
        flow_rate: Flow rate in mL/min
        pH: pH value
        column_temp: Column temperature in °C

    Returns:
        Plotly figure
    """
    # Extract time and %B values
    times = [point[0] for point in gradient]
    percentB = [point[1] for point in gradient]

    # Create figure
    fig = go.Figure()

    # Add gradient line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=percentB,
            mode="lines+markers",
            name="%B",
            line=dict(color="#0066cc", width=3),
            marker=dict(size=8),
        )
    )

    # Add other parameters as annotations
    annotations = [
        dict(
            x=max(times) * 0.8,
            y=max(percentB) * 0.9,
            text=f"Flow Rate: {flow_rate:.2f} mL/min<br>pH: {pH:.1f}<br>Temp: {column_temp:.1f}°C",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#0066cc",
            borderwidth=1,
            borderpad=4,
        )
    ]

    # Update layout
    fig.update_layout(
        title="Gradient Profile",
        xaxis_title="Time (min)",
        yaxis_title="%B",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=annotations,
        template="plotly_white",
    )

    return fig


def get_download_link(data, filename: str, text: str) -> str:
    """
    Generate a download link for data.

    Args:
        data: Data to download
        filename: Filename for the download
        text: Link text

    Returns:
        HTML for the download link
    """
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href


def plot_convergence(scores: List[float], best_scores: List[float]) -> go.Figure:
    """
    Create a plotly figure of the optimization convergence.

    Args:
        scores: List of scores for each trial
        best_scores: List of best scores up to each trial

    Returns:
        Plotly figure
    """
    trials = list(range(len(scores)))

    fig = go.Figure()

    # Add individual trial scores
    fig.add_trace(
        go.Scatter(
            x=trials,
            y=scores,
            mode="markers",
            name="Trial Score",
            marker=dict(size=8, color="#0066cc"),
        )
    )

    # Add best score line
    fig.add_trace(
        go.Scatter(
            x=trials,
            y=best_scores,
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

    return fig
