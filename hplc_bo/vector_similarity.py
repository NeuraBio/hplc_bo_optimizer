"""
Vector similarity module for HPLC method optimization.

This module provides functionality to:
1. Convert HPLC method parameters to vector representations
2. Pre-compute vectors for all validation runs
3. Find the most similar validation runs for new parameter sets
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from hplc_bo.data_types import ValidationResult


class VectorSimilarityEngine:
    """
    Engine for computing and storing vector representations of HPLC methods
    and finding similar methods using vector space similarity.
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        gradient_points: int = 20,
        validation_dir: Optional[str] = None,
    ):
        """
        Initialize the vector similarity engine.

        Args:
            cache_file: Path to the cache file for storing pre-computed vectors
            gradient_points: Number of points to use for gradient representation
            validation_dir: Directory containing validation results (optional)
        """
        # If validation_dir is provided, use it to construct default cache file path
        if validation_dir:
            self.cache_file = os.path.join(validation_dir, "vector_similarity_cache.json")
        else:
            self.cache_file = cache_file or os.path.join(
                "hplc_optimization", "validation", "vector_similarity_cache.json"
            )

        self.validation_vectors = []
        self.validation_results = []
        self.gradient_points = gradient_points
        # Base vector size will be:
        # - gradient_points (normalized gradient points)
        # - gradient_points-1 (slopes between points)
        # - 10 gradient features (statistics, patterns)
        # - 3 basic parameters (flow, pH, temp)
        # - 5 derived features (combinations of parameters)
        self.vector_size = gradient_points + (gradient_points - 1) + 10 + 3 + 5

    def _normalize_gradient(self, gradient, target_length=None, use_bo_anchors=True):
        """
        Normalize a gradient to a fixed length by interpolation.

        Args:
            gradient: The gradient table as a list of [time, %B] points
            target_length: The desired number of points (defaults to self.gradient_points)
            use_bo_anchors: If True, use the fixed anchor times from the BO system

        Returns:
            A list of %B values at evenly spaced time points
        """
        from hplc_bo.config import GRADIENT_ANCHOR_TIMES

        if target_length is None:
            target_length = self.gradient_points

        if not gradient:
            return [50.0] * target_length  # Default to 50% B if no gradient

        # Extract times and %B values
        try:
            # Handle different gradient formats
            if isinstance(gradient[0], dict):
                times = [point.get("time", i) for i, point in enumerate(gradient)]
                # Try both uppercase and lowercase keys for %B
                percentb = []
                for point in gradient:
                    if "%B" in point:
                        percentb.append(point["%B"])
                    elif "%b" in point:
                        percentb.append(point["%b"])
                    else:
                        percentb.append(50.0)  # Default if neither key exists
            else:
                times = [point[0] for point in gradient]
                percentb = [point[1] for point in gradient]

            # If we have only one point, duplicate it
            if len(times) == 1:
                times = [0, 1]
                percentb = [percentb[0], percentb[0]]

            # Determine target times for interpolation
            if use_bo_anchors and GRADIENT_ANCHOR_TIMES:
                # Use the fixed anchor times from the BO system
                normalized_times = GRADIENT_ANCHOR_TIMES
                target_length = len(GRADIENT_ANCHOR_TIMES)
            else:
                # Create evenly spaced time points
                max_time = max(times)
                if max_time == 0:
                    max_time = 1.0  # Avoid division by zero
                normalized_times = np.linspace(0, max_time, target_length)

            # Interpolate to get %B values at the target times
            normalized_percentb = np.interp(normalized_times, times, percentb)

            return normalized_percentb.tolist()

        except (TypeError, IndexError, ValueError) as e:
            print(f"Error normalizing gradient: {e}")
            return [50.0] * target_length  # Default to 50% B if error

    def _compute_gradient_features(self, gradient):
        """
        Compute additional features from the gradient profile.

        Args:
            gradient: Normalized gradient as a list of %B values

        Returns:
            List of gradient features (slope, curvature, etc.)
        """
        if not gradient or len(gradient) < 2:
            return [0] * 10  # Return 10 zero features

        # Calculate gradient statistics
        mean_b = np.mean(gradient)
        max_b = np.max(gradient)
        min_b = np.min(gradient)
        range_b = max_b - min_b
        std_b = np.std(gradient)
        median_b = np.median(gradient)

        # Calculate slope features
        slopes = np.diff(gradient)
        mean_slope = np.mean(slopes)
        max_slope = np.max(slopes)
        min_slope = np.min(slopes)
        slope_range = max_slope - min_slope

        # Calculate direction changes (pattern complexity)
        direction_changes = np.sum(np.diff(np.sign(slopes)) != 0)

        # Calculate curvature (second derivative)
        curvatures = np.diff(slopes) if len(slopes) > 1 else [0]
        mean_curvature = np.mean(curvatures)
        # max_curvature = np.max(np.abs(curvatures)) if len(curvatures) > 0 else 0

        # Calculate area under the curve (rough integral)
        # auc = np.sum(gradient) / len(gradient)

        # Calculate pattern features
        # - Is it mostly increasing?
        increasing_ratio = np.sum(slopes > 0) / len(slopes) if len(slopes) > 0 else 0.5
        # - Does it start high and end low?
        high_to_low = 1 if gradient[0] > gradient[-1] else 0
        # - Does it have a peak in the middle?
        has_peak = 1 if (np.argmax(gradient) > 0 and np.argmax(gradient) < len(gradient) - 1) else 0

        # Return all features
        return [
            mean_b / 100.0,  # Normalize to 0-1 range
            range_b / 100.0,  # Normalize to 0-1 range
            std_b / 50.0,  # Normalize to roughly 0-1 range
            median_b / 100.0,  # Normalize to 0-1 range
            mean_slope / 20.0,  # Normalize to roughly -1 to 1 range
            slope_range / 40.0,  # Normalize to roughly 0-1 range
            direction_changes / 5.0,  # Normalize assuming max 5 direction changes
            mean_curvature / 10.0,  # Normalize to roughly -1 to 1 range
            increasing_ratio,  # Already 0-1
            high_to_low + has_peak,  # 0, 1, or 2
        ]

    def params_to_vector(self, params: Dict[str, Any]) -> List[float]:
        """
        Convert HPLC method parameters to a vector representation.

        Args:
            params: Dictionary of HPLC method parameters

        Returns:
            Vector representation of the parameters
        """
        # Extract parameters with defaults
        flow_rate = params.get("flow_rate", 0.8)
        pH = params.get("pH", 7.0)
        column_temp = params.get("column_temp", 25.0)
        gradient = params.get("gradient", [])

        # Check if this is a BO-style parameter set with b0-b4 values
        is_bo_params = all(f"b{i}" in params for i in range(5))

        if is_bo_params:
            # Convert BO parameters to gradient format
            from hplc_bo.config import GRADIENT_ANCHOR_TIMES

            gradient = list(
                zip(
                    GRADIENT_ANCHOR_TIMES,
                    [params.get(f"b{i}", 50.0) for i in range(5)],
                    strict=False,
                )
            )

        # Normalize gradient to fixed length using BO anchor times
        normalized_gradient = self._normalize_gradient(gradient, use_bo_anchors=True)

        # Calculate gradient slopes (first derivative)
        gradient_slopes = (
            np.diff(normalized_gradient).tolist()
            if len(normalized_gradient) > 1
            else [0] * (self.gradient_points - 1)
        )

        # Compute additional gradient features
        gradient_features = self._compute_gradient_features(normalized_gradient)

        # Normalize basic parameters to 0-1 range
        # Flow rate typically 0.1-2.0 mL/min
        norm_flow = flow_rate / 2.0
        # pH typically 2-10
        norm_pH = (pH - 2.0) / 8.0 if 2.0 <= pH <= 10.0 else (0.0 if pH < 2.0 else 1.0)
        # Column temp typically 20-60°C
        norm_temp = (
            (column_temp - 20.0) / 40.0
            if 20.0 <= column_temp <= 60.0
            else (0.0 if column_temp < 20.0 else 1.0)
        )

        # Create derived features (combinations that might be important)
        # 1. Flow rate * temperature (affects pressure)
        flow_temp = norm_flow * norm_temp
        # 2. pH * temperature (affects selectivity)
        pH_temp = norm_pH * norm_temp
        # 3. Flow rate * pH (affects retention)
        flow_pH = norm_flow * norm_pH
        # 4. Average gradient slope * flow rate (affects separation)
        grad_flow = np.mean(np.abs(gradient_slopes)) * norm_flow if gradient_slopes else 0
        # 5. pH * gradient range (affects peak shape)
        pH_grad_range = (
            norm_pH * (np.max(normalized_gradient) - np.min(normalized_gradient)) / 100.0
            if normalized_gradient
            else 0
        )

        # Combine all features into a single vector
        vector = [
            # Basic parameters (normalized)
            norm_flow,
            norm_pH,
            norm_temp,
            # Normalized gradient points
            *[point / 100.0 for point in normalized_gradient],  # Normalize to 0-1 range
            # Gradient slopes
            *[slope / 20.0 for slope in gradient_slopes],  # Normalize to roughly -1 to 1 range
            # Additional gradient features
            *gradient_features,
            # Derived features
            flow_temp,
            pH_temp,
            flow_pH,
            grad_flow,
            pH_grad_range,
        ]

        return vector

    def validation_result_to_vector(self, result: ValidationResult) -> List[float]:
        """
        Convert a ValidationResult to a vector representation.

        Args:
            result: ValidationResult object

        Returns:
            Vector representation of the validation result
        """
        # Create a params dictionary from the validation result
        params = {
            "flow_rate": result.flow_rate,
            "pH": result.pH if hasattr(result, "pH") and result.pH is not None else 7.0,
            "column_temp": result.column_temperature,
            "gradient": result.gradient_table,
        }

        return self.params_to_vector(params)

    def precompute_validation_vectors(self, validation_results: List[ValidationResult]) -> None:
        """
        Pre-compute vector representations for all validation results.

        Args:
            validation_results: List of ValidationResult objects
        """
        self.validation_results = []
        self.validation_vectors = []

        for result in validation_results:
            # Skip invalid results
            if not hasattr(result, "score") or result.score is None or result.score <= -1000000000:
                continue

            # Convert to vector and store
            try:
                vector = self.validation_result_to_vector(result)
                self.validation_vectors.append(vector)
                self.validation_results.append(result)
            except Exception as e:
                print(f"Error converting validation result to vector: {e}")

        # Save to cache file
        self._save_cache()

        print(f"Pre-computed vectors for {len(self.validation_vectors)} validation results")

    def _save_cache(self) -> None:
        """Save the pre-computed vectors to the cache file."""
        cache_dir = os.path.dirname(self.cache_file)
        os.makedirs(cache_dir, exist_ok=True)

        # Convert validation results to serializable format
        serialized_results = []
        for result in self.validation_results:
            result_dict = {
                "filename": result.filename if hasattr(result, "filename") else "",
                "score": result.score if hasattr(result, "score") else 0.0,
                "flow_rate": result.flow_rate if hasattr(result, "flow_rate") else 0.0,
                "pH": result.pH if hasattr(result, "pH") else 7.0,
                "column_temperature": (
                    result.column_temperature if hasattr(result, "column_temperature") else 25.0
                ),
                "gradient_table": (
                    result.gradient_table if hasattr(result, "gradient_table") else []
                ),
                "injection_id": result.injection_id if hasattr(result, "injection_id") else 0,
                "result_id": result.result_id if hasattr(result, "result_id") else 0,
            }
            serialized_results.append(result_dict)

        # Convert numpy arrays to lists for JSON serialization
        serialized_vectors = [
            vector.tolist() if isinstance(vector, np.ndarray) else vector
            for vector in self.validation_vectors
        ]

        with open(self.cache_file, "w") as f:
            json.dump({"vectors": serialized_vectors, "results": serialized_results}, f, indent=2)

        print(f"Saved vector similarity cache to {self.cache_file}")

    def load_cache(self) -> bool:
        """
        Load pre-computed vectors from the cache file.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not os.path.exists(self.cache_file):
            print(f"Cache file {self.cache_file} not found")
            return False

        try:
            with open(self.cache_file, "r") as f:
                cache = json.load(f)

                # Convert lists back to numpy arrays for vectors
                self.validation_vectors = [np.array(vector) for vector in cache["vectors"]]

                # Import ValidationResult here to avoid circular imports
                from hplc_bo.validation import ValidationResult

                self.validation_results = []

                for result_dict in cache["results"]:
                    # Create ValidationResult object
                    result = ValidationResult(
                        pdf_path=result_dict.get("pdf_path", ""),
                        filename=result_dict.get("filename", ""),
                        rt_list=result_dict.get("rt_list", []),
                        peak_widths=result_dict.get("peak_widths", []),
                        tailing_factors=result_dict.get("tailing_factors", []),
                        column_temperature=result_dict.get("column_temperature"),
                        flow_rate=result_dict.get("flow_rate"),
                        solvent_a=result_dict.get("solvent_a"),
                        solvent_b=result_dict.get("solvent_b"),
                        gradient_table=result_dict.get("gradient_table", []),
                        score=result_dict.get("score"),
                        chemist_rating=result_dict.get("chemist_rating"),
                        notes=result_dict.get("notes"),
                        rt_table_data=result_dict.get("rt_table_data"),
                        areas=result_dict.get("areas"),
                        plate_counts=result_dict.get("plate_counts"),
                        injection_id=result_dict.get("injection_id"),
                        result_id=result_dict.get("result_id"),
                        sample_set_id=result_dict.get("sample_set_id"),
                    )
                    self.validation_results.append(result)

            print(f"Loaded {len(self.validation_vectors)} vectors from cache")
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def find_similar_runs(
        self,
        params: Dict[str, Any],
        top_k: int = 5,
        metric: str = "cosine",
        score_threshold: float = None,
    ) -> List[Tuple[ValidationResult, float]]:
        """
        Find the most similar validation runs to the given parameters.

        Args:
            params: Dictionary of HPLC method parameters
            top_k: Number of similar runs to return
            metric: Distance metric to use ('euclidean', 'cosine', 'manhattan', 'correlation')
            score_threshold: Optional minimum score threshold to filter results

        Returns:
            List of (ValidationResult, similarity_score) tuples
        """
        if not self.validation_vectors:
            if not self.load_cache():
                print("No pre-computed vectors available")
                return []

        # Convert params to vector
        query_vector = self.params_to_vector(params)

        # Compute distances to all validation vectors
        distances = cdist([query_vector], self.validation_vectors, metric=metric)[0]

        # Create a list of (index, distance, score) tuples
        results = [
            (i, distances[i], self.validation_results[i].score) for i in range(len(distances))
        ]

        # Filter by score threshold if provided
        if score_threshold is not None:
            filtered_results = [(i, d, s) for i, d, s in results if s >= score_threshold]

            if filtered_results:
                results = filtered_results
            else:
                print(f"Warning: No results found with score >= {score_threshold}")
                # Continue with all results

        # Convert distances to similarity scores based on the metric
        if metric == "cosine" or metric == "correlation":
            # For cosine and correlation, distance is already in [0,2] range
            # Convert to similarity: 1 = identical, 0 = completely different
            similarities = [(i, 1 - d / 2, s) for i, d, s in results]
        else:
            # For euclidean and other metrics, convert to similarity score
            # using a decay function that gives higher weight to closer matches
            similarities = [(i, 1 / (1 + d), s) for i, d, s in results]

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top_k results
        top_results = similarities[:top_k]

        # Return the results with similarity scores
        return [(self.validation_results[i], sim) for i, sim, _ in top_results]

    def find_closest_run(
        self,
        params: Dict[str, Any],
        metric: str = "cosine",
        score_threshold: float = None,
        best_score_bias: bool = False,
    ) -> Tuple[ValidationResult, float]:
        """
        Find the closest validation run to the given parameters.

        Args:
            params: Dictionary of HPLC method parameters
            metric: Distance metric to use ('euclidean', 'cosine', 'manhattan', 'correlation')
            score_threshold: Optional minimum score threshold to filter results
            best_score_bias: If True, apply a bias toward higher-scoring runs

        Returns:
            Tuple of (closest ValidationResult, similarity_score)
        """
        # If we want to bias toward best scores, first try with a high threshold
        if best_score_bias and score_threshold is None:
            # Try with a high score threshold first
            best_threshold = -15.0  # Adjust based on your score distribution
            best_matches = self.find_similar_runs(
                params, top_k=3, metric=metric, score_threshold=best_threshold
            )

            if best_matches:
                # Return the closest of the high-scoring runs
                return best_matches[0]

        # Fall back to regular similarity search
        similar_runs = self.find_similar_runs(
            params, top_k=1, metric=metric, score_threshold=score_threshold
        )

        if not similar_runs:
            # If no runs found (possibly due to score threshold), try without threshold
            if score_threshold is not None:
                similar_runs = self.find_similar_runs(params, top_k=1, metric=metric)

            if not similar_runs:
                raise ValueError("No similar runs found")

        return similar_runs[0]


def precompute_vectors(
    validation_results: List[ValidationResult], cache_file: Optional[str] = None
) -> str:
    """
    Pre-compute vector representations for all validation results and save to cache.

    Args:
        validation_results: List of ValidationResult objects
        cache_file: Path to the cache file (optional)

    Returns:
        Path to the cache file
    """
    engine = VectorSimilarityEngine(cache_file)
    engine.precompute_validation_vectors(validation_results)
    return engine.cache_file
