# HPLC Method Optimizer UI Design Plan

## UI Vision and Goals

### Primary Users
- **Lab Chemists**: Need intuitive interfaces with minimal technical complexity
- **Power Users/Developers**: Need access to advanced features and observability tools

### Key UI Goals
1. **Intuitive & Accessible**: Simple workflow for chemists without technical expertise
2. **Visually Appealing**: Clean, modern design with clear data visualization
3. **Robust**: Stable import structure and error handling
4. **Modular**: Well-organized components for maintainability
5. **Observability**: Advanced monitoring for developers

## Architecture Plan

### 1. Application Structure

```
hplc_bo_optimizer/
├── app/                      # Streamlit app code
│   ├── __init__.py           # Make app a proper package
│   ├── main.py               # Main entry point for Streamlit
│   ├── config.py             # Configuration management
│   ├── components/           # UI components
│   │   ├── __init__.py
│   │   ├── navigation.py     # Navigation sidebar
│   │   ├── validation.py     # Validation view
│   │   ├── simulation.py     # Simulation view
│   │   ├── suggestion.py     # Parameter suggestion view
│   │   ├── reporting.py      # Results reporting view
│   │   ├── status.py         # Experiment status view
│   │   ├── advanced.py       # Advanced features for power users
│   │   └── monitoring.py     # System monitoring for developers
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── session.py        # Session state management
│   │   ├── ui_helpers.py     # UI helper functions
│   │   ├── file_handlers.py  # File upload/download helpers
│   │   └── visualization.py  # Data visualization helpers
│   └── services/             # Service layer to interact with backend
│       ├── __init__.py
│       ├── optimizer.py      # Interface to hplc_optimize.py functionality
│       ├── data_service.py   # Data loading/saving operations
│       └── process_manager.py # Background process management
└── streamlit_app.py          # Top-level entry script
```

### 2. Import Strategy

To avoid the import issues we encountered:

1. Use absolute imports throughout the application
2. Set `PYTHONPATH=/app` in Docker to ensure proper package resolution
3. Use a single entry point (`streamlit_app.py`) that imports from the app package
4. Ensure all directories have `__init__.py` files to be recognized as packages

### 3. Service Layer

Create a service layer that wraps the CLI functionality to make it accessible to the UI:

```python
# app/services/optimizer.py
import subprocess
import json
import os
from typing import Dict, Any, List, Optional

class OptimizerService:
    """Service to interact with the HPLC optimizer CLI"""
    
    def __init__(self, client_lab: str, experiment: str, output_dir: str = "hplc_optimization"):
        self.client_lab = client_lab
        self.experiment = experiment
        self.output_dir = output_dir
        
    def validate(self, pdf_dir: str) -> Dict[str, Any]:
        """Run validation on historical PDF data"""
        # Implementation that calls the CLI command
        
    def simulate(self, n_trials: int = 10) -> Dict[str, Any]:
        """Run BO simulation"""
        # Implementation that calls the CLI command
        
    def suggest(self) -> Dict[str, Any]:
        """Get suggestion for next trial"""
        # Implementation that calls the CLI command
        
    def report(self, trial_id: int, rt_file: str, gradient_file: Optional[str] = None) -> Dict[str, Any]:
        """Report results for a trial"""
        # Implementation that calls the CLI command
```

## UI Workflow and Features

### 1. Home/Dashboard View
- Welcome message and project overview
- Quick stats on current experiment status
- Cards for recent activities and next steps
- Getting started guide for new users

### 2. Validation View
- PDF upload area with drag-and-drop support
- Progress indicator for validation process
- Results summary with key metrics
- Interactive visualizations of validation data
- Option to download validation report

### 3. Simulation View
- Controls to configure simulation parameters
- Run simulation button with progress indicator
- Interactive plots showing simulation results
- Efficiency metrics compared to manual experimentation
- Option to download simulation report

### 4. Suggestion View
- Display of suggested parameters for next trial
- Visual representation of gradient profile
- Option to adjust parameters manually if needed
- Export suggestion as PDF/CSV for lab use
- Button to mark suggestion as "in progress"

### 5. Reporting View
- Form to upload result files (retention times, gradients)
- Trial selection dropdown
- Data validation and preview
- Submit button to record results
- Success confirmation with updated metrics

### 6. Status View
- Overview of all trials with status indicators
- Timeline visualization of experiment progress
- Performance metrics and convergence plots
- Data table with sortable/filterable results

### 7. Advanced View (for power users)
- Direct access to optimizer parameters
- Custom simulation configurations
- Batch processing options
- Export raw data in various formats

### 8. Monitoring View (for developers)
- System health metrics
- Process monitoring
- Log viewer
- Configuration inspector
- Performance profiling

## Implementation Strategy

### Phase 1: Core Framework
1. Set up proper project structure with correct imports
2. Create service layer to interface with CLI
3. Implement basic navigation and layout
4. Build session state management

### Phase 2: Essential Features
1. Implement validation workflow
2. Implement simulation workflow
3. Implement suggestion generation
4. Implement basic reporting

### Phase 3: Enhanced UI and Visualization
1. Add interactive data visualizations
2. Improve UI styling and responsiveness
3. Implement file upload/download functionality
4. Add progress indicators and notifications

### Phase 4: Advanced Features
1. Implement status dashboard
2. Add advanced configuration options
3. Implement monitoring tools
4. Add batch processing capabilities

## Technical Considerations

### Docker Integration
- Ensure Streamlit app can access CLI functionality in Docker
- Set proper environment variables for import resolution
- Mount volumes correctly for file access
- Use Docker networking for service communication

### Error Handling
- Implement robust error handling throughout the app
- Provide user-friendly error messages
- Log detailed errors for debugging
- Add retry mechanisms for transient failures

### State Management
- Use Streamlit session state for persistent data
- Implement proper state initialization
- Handle concurrent users appropriately
- Cache expensive computations

### Performance Optimization
- Optimize file processing for large datasets
- Implement background processing for long-running tasks
- Use caching for repeated operations
- Monitor and optimize memory usage

## Next Steps

1. **Implement Core Framework**
   - Create proper directory structure with `__init__.py` files
   - Set up service layer to interface with CLI
   - Implement basic navigation and layout

2. **Build Essential Features**
   - Start with validation workflow as it's the first step
   - Then implement simulation, suggestion, and reporting

3. **Test and Iterate**
   - Get feedback from lab chemists
   - Refine UI based on user experience
   - Address any performance or stability issues
