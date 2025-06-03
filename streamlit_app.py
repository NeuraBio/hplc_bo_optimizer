#!/usr/bin/env python
"""
HPLC Method Optimizer Streamlit App

This is the top-level entry script for the Streamlit application.
It imports the main app module and runs it.
"""

import os
import sys

# Ensure app directory is in the Python path
# This is important for Docker where PYTHONPATH=/app
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main app module
from app.main import main

if __name__ == "__main__":
    main()
