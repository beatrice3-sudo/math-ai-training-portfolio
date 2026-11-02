# Computational Mathematics - AI Training Portfolio

This repository demonstrates end-to-end workflows for advanced mathematical problem-solving: **translating math → executable code → numerical validation.**

Portfolio for Senior Mathematics Expert (USD 80-120/hr) - Advanced Math to AI Training Workflows

### Skills Demonstrated
**Math Areas:** Numerical Analysis, Optimization, Probability & Statistics, Symbolic Math, Linear Programming
**Python Stack:** NumPy, SciPy, SymPy, pandas, Matplotlib, Jupyter, statsmodels, CVXPY (via SciPy linprog)
**Workflows:** CLI-based, reproducible, edge case testing, parameter sweeps, symbolic vs numerical validation

### 01 - Numerical Optimization & Validation
- Problem: Find global minimum of f(x)=x^4-3x^3+2x
- Workflow: Symbolic derivative with SymPy → bounded optimization with SciPy → derivative check → parameter sweep
- File: `01_numerical_optimization.py`

### 02 - Monte Carlo Validation of Central Limit Theorem
- Problem: Validate CLT for Exponential(2) distribution
- Workflow: 10,000 Monte Carlo simulations → observed vs theoretical comparison → normality test → visualization
- File: `02_monte_carlo_CLT.py`

### 03 - Linear Programming & Feasible Region Analysis
- Problem: Maximize 3x+5y subject to x+2y≤8, 3x+2y≤12
- Workflow: SciPy linprog (CVXPY-equivalent) → symbolic intersection → constraint validation → sensitivity sweep
- File: `03_linear_optimization.py`

### How to Run
```bash
python 01_numerical_optimization.py
python 02_monte_carlo_CLT.py
python 03_linear_optimization.py
