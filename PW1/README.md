## Lab B - Automated Decay Analysis Pipeline

This section implements an automated data processing and visualization pipeline for radioactive decay data using Python and Snakemake.

### Pipeline Steps
1. **Data Input**: Reads observed decay counts from `decay_observed.csv`.
2. **Analytical Modeling**: Calculates theoretical decay values using $N(t) = N_0 \cdot e^{-\lambda t}$ with $\lambda = 0.3$.
3. **Visualization**: Generates a 1x2 subplot comparing observed data against the analytical model curve.
4. **Output**: Saves the final plot automatically as `figure.png`.

### Files Included
- `decay_observed.csv`: Input dataset containing time and observed count values.
- `plot.py`: Python script processing data and rendering the figures.
- `Snakefile`: Snakemake workflow definition specifying input/output rules.
- `figure.png`: Output graph created by the pipeline.

### How to Run
To run the automated workflow, navigate to the `Lab B` directory inside your active environment and execute Snakemake: