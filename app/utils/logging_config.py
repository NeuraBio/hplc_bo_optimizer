"""
Logging Configuration Module

This module provides a centralized logging configuration for the HPLC Method Optimizer application.
It sets up logging to both stdout and a file, with rotation, to ensure logs are available for debugging
and can be viewed with docker logs.
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def configure_logging(app_name="hplc_optimizer", log_level=logging.INFO):
    """
    Configure logging for the application.

    Args:
        app_name: Name of the application (used for log file naming)
        log_level: Logging level (default: INFO)

    Returns:
        Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    # Set up log file path with rotation
    log_file = log_dir / f"{app_name}.log"

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",  # 10 MB
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log startup message
    logger.info(f"Logging configured for {app_name} (level: {logging.getLevelName(log_level)})")
    logger.info(f"Log file: {log_file}")

    return logger
