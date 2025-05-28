# HPLC Bayesian Optimization Project TODOs

This file tracks the development tasks for the HPLC Bayesian Optimization project.

1.  **PDF Data Parsing:**
    *   Develop Python code (e.g., using `pdfplumber`) to extract necessary information from PDF lab reports.
    *   Data to extract:
        *   RT table: RT, TailingFactor, PlateCount(N), Peak Height (to approximate width at base).
        *   "Target Column Temperature".
        *   "Solvent A Name" and "Solvent B Name".
        *   "Gradient Table" data.

2.  **Gradient Translation & Fortification:**
    *   Adapt the parsed "Gradient Table" from PDFs to be compatible with the existing `interpolate_gradient` and `expand_gradient` functions, which utilize anchored gradient points.
    *   Confirm and strengthen this approach for representing gradients as Bayesian Optimization parameters.

3.  **Validation with Real Trial Runs:**
    *   Utilize known historical lab trial runs (after parsing them from PDFs) to:
        *   Validate the `compute_score` function: Ensure it assigns the best score to the chromatogram identified as optimal by chemists.
        *   Assess score convergence behavior during the optimization process.

4.  **Write Unit Tests:**
    *   Create comprehensive unit tests for:
        *   The PDF parsing module/functions.
        *   The gradient translation logic.
        *   The validation process for `compute_score` using known good/bad runs.

5.  **Add `pdfplumber` dependency:**
    *   Use `poetry add pdfplumber` to include the library in the project's dependencies.

6.  - Ignore rows from gradient table once FlowRate changes 
    - Ignore rows from RT table with no Area or area under 500,000
    - Report should have columns from RT tabel
    - Report should have compute_score inputs e.g. peak_width