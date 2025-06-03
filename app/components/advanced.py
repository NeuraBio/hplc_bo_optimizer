"""
Advanced Component

This module provides the advanced view for the HPLC Method Optimizer Streamlit app.
This view is intended for power users who need more control over the optimization process.
"""

import json

import streamlit as st
import yaml

from app.config import get_config, save_config
from app.services.optimizer import OptimizerService
from app.utils.session import get_session_value
from app.utils.ui_helpers import error_box, main_header, section_header, success_box, warning_box


def render_advanced_view():
    """
    Render the advanced view.
    """
    main_header("Advanced Configuration")

    st.markdown(
        """
    This page provides advanced configuration options for power users.
    These settings allow fine-tuning of the optimization process, parameter spaces, and scoring functions.
    """
    )

    # Get configuration from session state
    client_lab = get_session_value("client_lab", "NuraxsDemo")
    experiment = get_session_value("experiment", "HPLC-Optimization")
    output_dir = get_session_value("output_dir", "hplc_optimization")

    # Initialize optimizer service
    optimizer_service = OptimizerService(client_lab, experiment, output_dir)

    # Create tabs for different advanced options
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Optimizer Settings", "Parameter Space", "Scoring Function", "Export/Import"]
    )

    with tab1:
        render_optimizer_settings(optimizer_service)

    with tab2:
        render_parameter_space(optimizer_service)

    with tab3:
        render_scoring_function(optimizer_service)

    with tab4:
        render_export_import(optimizer_service)


def render_optimizer_settings(optimizer_service: OptimizerService):
    """
    Render the optimizer settings section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Optimizer Settings")

    # Get current config
    config = get_config()
    optimizer_config = config.get("optimizer", {})

    st.write(
        """
    These settings control the behavior of the Bayesian Optimization process.
    Adjust these parameters to fine-tune the optimization strategy.
    """
    )

    # Optimizer type
    optimizer_type = st.selectbox(
        "Optimizer Type",
        options=["optuna", "skopt"],
        index=0 if optimizer_config.get("type", "optuna") == "optuna" else 1,
        help="Optimization framework to use",
    )

    # Number of trials
    n_trials = st.slider(
        "Default Number of Trials",
        min_value=5,
        max_value=100,
        value=optimizer_config.get("n_trials", 20),
        step=5,
        help="Default number of trials for optimization",
    )

    # Sampler
    sampler = st.selectbox(
        "Sampler",
        options=["TPE", "CMA-ES", "Random", "Grid", "NSGA-II"],
        index=["TPE", "CMA-ES", "Random", "Grid", "NSGA-II"].index(
            optimizer_config.get("sampler", "TPE")
        ),
        help="Sampling algorithm for the optimization process",
    )

    # Pruner
    pruner = st.selectbox(
        "Pruner",
        options=["None", "Median", "Percentile", "Hyperband"],
        index=["None", "Median", "Percentile", "Hyperband"].index(
            optimizer_config.get("pruner", "None")
        ),
        help="Pruning algorithm for early stopping of unpromising trials",
    )

    # Advanced options expander
    with st.expander("Advanced Optimizer Options", expanded=False):
        # Multivariate
        multivariate = st.checkbox(
            "Multivariate Optimization",
            value=optimizer_config.get("multivariate", False),
            help="Use multivariate optimization (multiple objectives)",
        )

        # Direction
        direction = st.selectbox(
            "Direction",
            options=["maximize", "minimize"],
            index=0 if optimizer_config.get("direction", "maximize") == "maximize" else 1,
            help="Whether to maximize or minimize the objective function",
        )

        # Seed
        seed = st.number_input(
            "Random Seed",
            value=optimizer_config.get("seed", 42),
            min_value=0,
            max_value=10000,
            step=1,
            help="Random seed for reproducibility",
        )

        # Timeout
        timeout = st.number_input(
            "Timeout (seconds)",
            value=optimizer_config.get("timeout", 600),
            min_value=0,
            max_value=3600,
            step=60,
            help="Maximum time for optimization in seconds (0 = no timeout)",
        )

    # Save button
    if st.button("Save Optimizer Settings"):
        # Update config
        config["optimizer"] = {
            "type": optimizer_type,
            "n_trials": n_trials,
            "sampler": sampler,
            "pruner": pruner,
            "multivariate": multivariate if "multivariate" in locals() else False,
            "direction": direction if "direction" in locals() else "maximize",
            "seed": seed if "seed" in locals() else 42,
            "timeout": timeout if "timeout" in locals() else 600,
        }

        # Save config
        save_config(config)

        # Show success message
        success_box("Optimizer settings saved successfully!")


def render_parameter_space(optimizer_service: OptimizerService):
    """
    Render the parameter space section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Parameter Space Configuration")

    # Get current config
    config = get_config()
    param_space = config.get("parameter_space", {})

    st.write(
        """
    Define the parameter space for optimization.
    These settings control the range of values that the optimizer can explore.
    """
    )

    # pH range
    st.write("### pH Range")

    col1, col2 = st.columns(2)

    with col1:
        ph_min = st.number_input(
            "Minimum pH",
            value=param_space.get("ph_min", 2.0),
            min_value=1.0,
            max_value=12.0,
            step=0.1,
            help="Minimum pH value",
        )

    with col2:
        ph_max = st.number_input(
            "Maximum pH",
            value=param_space.get("ph_max", 10.0),
            min_value=1.0,
            max_value=12.0,
            step=0.1,
            help="Maximum pH value",
        )

    # Temperature range
    st.write("### Temperature Range")

    col1, col2 = st.columns(2)

    with col1:
        temp_min = st.number_input(
            "Minimum Temperature (°C)",
            value=param_space.get("temp_min", 25.0),
            min_value=20.0,
            max_value=80.0,
            step=1.0,
            help="Minimum column temperature",
        )

    with col2:
        temp_max = st.number_input(
            "Maximum Temperature (°C)",
            value=param_space.get("temp_max", 45.0),
            min_value=20.0,
            max_value=80.0,
            step=1.0,
            help="Maximum column temperature",
        )

    # Flow rate range
    st.write("### Flow Rate Range")

    col1, col2 = st.columns(2)

    with col1:
        flow_min = st.number_input(
            "Minimum Flow Rate (mL/min)",
            value=param_space.get("flow_min", 0.5),
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            help="Minimum flow rate",
        )

    with col2:
        flow_max = st.number_input(
            "Maximum Flow Rate (mL/min)",
            value=param_space.get("flow_max", 2.0),
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            help="Maximum flow rate",
        )

    # Gradient parameters
    st.write("### Gradient Parameters")

    # Number of gradient segments
    n_segments = st.slider(
        "Number of Gradient Segments",
        min_value=2,
        max_value=10,
        value=param_space.get("n_segments", 4),
        step=1,
        help="Number of segments in the gradient profile",
    )

    # Gradient range
    col1, col2 = st.columns(2)

    with col1:
        gradient_min = st.number_input(
            "Minimum %B",
            value=param_space.get("gradient_min", 5.0),
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help="Minimum %B in gradient",
        )

    with col2:
        gradient_max = st.number_input(
            "Maximum %B",
            value=param_space.get("gradient_max", 95.0),
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help="Maximum %B in gradient",
        )

    # Run time
    run_time = st.slider(
        "Maximum Run Time (min)",
        min_value=5,
        max_value=60,
        value=param_space.get("run_time", 30),
        step=5,
        help="Maximum run time for the method",
    )

    # Advanced options expander
    with st.expander("Advanced Parameter Options", expanded=False):
        # Discrete pH values
        discrete_ph = st.checkbox(
            "Use Discrete pH Values",
            value=param_space.get("discrete_ph", False),
            help="Use discrete pH values instead of continuous range",
        )

        if discrete_ph:
            ph_values = st.text_input(
                "Discrete pH Values (comma-separated)",
                value=",".join(
                    map(
                        str,
                        param_space.get(
                            "ph_values", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                        ),
                    )
                ),
                help="Comma-separated list of discrete pH values",
            )

        # Discrete temperature values
        discrete_temp = st.checkbox(
            "Use Discrete Temperature Values",
            value=param_space.get("discrete_temp", False),
            help="Use discrete temperature values instead of continuous range",
        )

        if discrete_temp:
            temp_values = st.text_input(
                "Discrete Temperature Values (comma-separated)",
                value=",".join(
                    map(str, param_space.get("temp_values", [25.0, 30.0, 35.0, 40.0, 45.0]))
                ),
                help="Comma-separated list of discrete temperature values",
            )

        # Gradient constraints
        enforce_gradient_constraints = st.checkbox(
            "Enforce Gradient Constraints",
            value=param_space.get("enforce_gradient_constraints", True),
            help="Enforce constraints on gradient profile (e.g., monotonically increasing)",
        )

    # Save button
    if st.button("Save Parameter Space"):
        # Validate inputs
        if ph_min >= ph_max:
            error_box("Minimum pH must be less than maximum pH.")
            return

        if temp_min >= temp_max:
            error_box("Minimum temperature must be less than maximum temperature.")
            return

        if flow_min >= flow_max:
            error_box("Minimum flow rate must be less than maximum flow rate.")
            return

        if gradient_min >= gradient_max:
            error_box("Minimum %B must be less than maximum %B.")
            return

        # Update config
        config["parameter_space"] = {
            "ph_min": ph_min,
            "ph_max": ph_max,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "flow_min": flow_min,
            "flow_max": flow_max,
            "n_segments": n_segments,
            "gradient_min": gradient_min,
            "gradient_max": gradient_max,
            "run_time": run_time,
            "discrete_ph": discrete_ph if "discrete_ph" in locals() else False,
            "discrete_temp": discrete_temp if "discrete_temp" in locals() else False,
            "enforce_gradient_constraints": (
                enforce_gradient_constraints if "enforce_gradient_constraints" in locals() else True
            ),
        }

        # Add discrete values if specified
        if discrete_ph and "ph_values" in locals():
            try:
                ph_values_list = [float(x.strip()) for x in ph_values.split(",")]
                config["parameter_space"]["ph_values"] = ph_values_list
            except ValueError:
                error_box("Invalid pH values. Please enter comma-separated numbers.")
                return

        if discrete_temp and "temp_values" in locals():
            try:
                temp_values_list = [float(x.strip()) for x in temp_values.split(",")]
                config["parameter_space"]["temp_values"] = temp_values_list
            except ValueError:
                error_box("Invalid temperature values. Please enter comma-separated numbers.")
                return

        # Save config
        save_config(config)

        # Show success message
        success_box("Parameter space saved successfully!")


def render_scoring_function(optimizer_service: OptimizerService):
    """
    Render the scoring function section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Scoring Function Configuration")

    # Get current config
    config = get_config()
    scoring_config = config.get("scoring", {})

    st.write(
        """
    Configure the scoring function used to evaluate HPLC methods.
    These settings control how different aspects of chromatogram quality are weighted.
    """
    )

    # Scoring weights
    st.write("### Scoring Weights")

    # Resolution weight
    resolution_weight = st.slider(
        "Resolution Weight",
        min_value=0.0,
        max_value=1.0,
        value=scoring_config.get("resolution_weight", 0.4),
        step=0.1,
        help="Weight for peak resolution in the scoring function",
    )

    # Tailing factor weight
    tailing_weight = st.slider(
        "Tailing Factor Weight",
        min_value=0.0,
        max_value=1.0,
        value=scoring_config.get("tailing_weight", 0.2),
        step=0.1,
        help="Weight for peak tailing factor in the scoring function",
    )

    # Plate count weight
    plates_weight = st.slider(
        "Plate Count Weight",
        min_value=0.0,
        max_value=1.0,
        value=scoring_config.get("plates_weight", 0.2),
        step=0.1,
        help="Weight for theoretical plate count in the scoring function",
    )

    # Run time weight
    runtime_weight = st.slider(
        "Run Time Weight",
        min_value=0.0,
        max_value=1.0,
        value=scoring_config.get("runtime_weight", 0.1),
        step=0.1,
        help="Weight for method run time in the scoring function",
    )

    # Peak capacity weight
    capacity_weight = st.slider(
        "Peak Capacity Weight",
        min_value=0.0,
        max_value=1.0,
        value=scoring_config.get("capacity_weight", 0.1),
        step=0.1,
        help="Weight for peak capacity in the scoring function",
    )

    # Advanced options expander
    with st.expander("Advanced Scoring Options", expanded=False):
        # Minimum resolution
        min_resolution = st.number_input(
            "Minimum Resolution",
            value=scoring_config.get("min_resolution", 1.5),
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            help="Minimum acceptable resolution between peaks",
        )

        # Target tailing factor
        target_tailing = st.number_input(
            "Target Tailing Factor",
            value=scoring_config.get("target_tailing", 1.0),
            min_value=0.5,
            max_value=2.0,
            step=0.1,
            help="Target tailing factor (1.0 = symmetric peak)",
        )

        # Minimum plate count
        min_plates = st.number_input(
            "Minimum Plate Count",
            value=scoring_config.get("min_plates", 2000),
            min_value=0,
            max_value=100000,
            step=1000,
            help="Minimum acceptable theoretical plate count",
        )

        # Use custom scoring function
        use_custom_scoring = st.checkbox(
            "Use Custom Scoring Function",
            value=scoring_config.get("use_custom_scoring", False),
            help="Use a custom scoring function defined in a Python file",
        )

        if use_custom_scoring:
            custom_scoring_path = st.text_input(
                "Custom Scoring Function Path",
                value=scoring_config.get("custom_scoring_path", ""),
                help="Path to Python file containing custom scoring function",
            )

    # Total weight check
    total_weight = (
        resolution_weight + tailing_weight + plates_weight + runtime_weight + capacity_weight
    )

    if abs(total_weight - 1.0) > 0.001:
        warning_box(
            f"Total weight ({total_weight:.1f}) is not equal to 1.0. Consider adjusting weights."
        )

    # Save button
    if st.button("Save Scoring Configuration"):
        # Update config
        config["scoring"] = {
            "resolution_weight": resolution_weight,
            "tailing_weight": tailing_weight,
            "plates_weight": plates_weight,
            "runtime_weight": runtime_weight,
            "capacity_weight": capacity_weight,
            "min_resolution": min_resolution if "min_resolution" in locals() else 1.5,
            "target_tailing": target_tailing if "target_tailing" in locals() else 1.0,
            "min_plates": min_plates if "min_plates" in locals() else 2000,
            "use_custom_scoring": use_custom_scoring if "use_custom_scoring" in locals() else False,
        }

        # Add custom scoring path if specified
        if use_custom_scoring and "custom_scoring_path" in locals():
            config["scoring"]["custom_scoring_path"] = custom_scoring_path

        # Save config
        save_config(config)

        # Show success message
        success_box("Scoring configuration saved successfully!")


def render_export_import(optimizer_service: OptimizerService):
    """
    Render the export/import section.

    Args:
        optimizer_service: Optimizer service instance
    """
    section_header("Export/Import Configuration")

    st.write(
        """
    Export the current configuration to a file or import configuration from a file.
    This allows you to save and share optimization settings between experiments.
    """
    )

    # Export configuration
    st.write("### Export Configuration")

    if st.button("Export Configuration"):
        # Get current config
        config = get_config()

        # Convert to YAML
        yaml_str = yaml.dump(config, default_flow_style=False)

        # Provide download link
        st.download_button(
            "Download YAML Config",
            yaml_str,
            "hplc_optimizer_config.yaml",
            "text/yaml",
            key="download-config-yaml",
        )

    # Import configuration
    st.write("### Import Configuration")

    uploaded_file = st.file_uploader(
        "Upload Configuration File",
        type=["yaml", "yml", "json"],
        help="Upload a YAML or JSON configuration file",
    )

    if uploaded_file is not None:
        try:
            # Determine file type
            if uploaded_file.name.endswith((".yaml", ".yml")):
                # Parse YAML
                config = yaml.safe_load(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                # Parse JSON
                config = json.load(uploaded_file)
            else:
                error_box("Unsupported file format. Please upload a YAML or JSON file.")
                return

            # Validate config
            if not isinstance(config, dict):
                error_box("Invalid configuration format. Configuration must be a dictionary.")
                return

            # Display preview
            st.write("#### Configuration Preview")
            st.json(config)

            # Import button
            if st.button("Import Configuration"):
                # Save config
                save_config(config)

                # Show success message
                success_box("Configuration imported successfully!")
        except Exception as e:
            error_box(f"Error importing configuration: {str(e)}")

    # Reset configuration
    st.write("### Reset Configuration")

    if st.button("Reset to Default Configuration"):
        # Confirm reset
        st.write(
            "Are you sure you want to reset to the default configuration? This action cannot be undone."
        )

        if st.button("Yes, Reset Configuration"):
            # Reset config
            from app.config import DEFAULT_CONFIG

            save_config(DEFAULT_CONFIG)

            # Show success message
            success_box("Configuration reset to defaults successfully!")
