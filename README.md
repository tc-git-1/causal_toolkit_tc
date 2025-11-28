# causal_toolkit_tc
[![Tests](https://github.com/tc-git-1/causal_toolkit_tc/workflows/Tests/badge.svg)](https://github.com/tc-git-1/causal_toolkit_tc/actions)
[![PyPI version](https://badge.fury.io/py/causal_toolkit_tc.svg)](https://pypi.org/project/causal_toolkit_tc/)

A Python package for causal inference methods including ATE estimation, propensity scores, and meta-learners.

---

## 📦 Features
- **RCT Methods**: `calculate_ate_ci()`, `calculate_ate_pvalue()`
- **Propensity Score**: `ipw()`, `doubly_robust()`
- **Meta-Learners**: `s_learner_discrete()`, `t_learner_discrete()`, `x_learner_discrete()`, `double_ml_cate()`

---

## ✅ Installation

**For Users:**
```bash
pip install causal_toolkit_tc
```

**For Contributors:**
```bash
git clone https://github.com/tc-git-1/causal_toolkit_tc.git
cd causal_toolkit_tc
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

Here are examples for every function in the package.

### Data Generation
First, let's create a sample dataset that simulates an observational study (with confounders) and a continuous treatment dataset for Double ML.

```python
import pandas as pd
import numpy as np
from causal_toolkit_tc.rct import calculate_ate_ci, calculate_ate_pvalue
from causal_toolkit_tc.propensity import ipw, doubly_robust
from causal_toolkit_tc.meta_learners import (
    s_learner_discrete, t_learner_discrete, x_learner_discrete, double_ml_cate
)

# 1. Generate Binary Treatment Data (for RCT, Propensity, S/T/X Learners)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.normal(30, 10, n),
    'income': np.random.normal(50000, 10000, n),
    'T': np.random.binomial(1, 0.5, n), # Binary Treatment
})

# Outcome depends on T and confounders
# True ATE = 2.0
df['Y'] = 2*df['T'] + 0.05*df['age'] + 0.001*df['income'] + np.random.normal(0, 1, n)

# 2. Generate Continuous Treatment Data (for Double ML)
df_cont = df.copy()
df_cont['T_cont'] = np.random.normal(5, 2, n) # Continuous Treatment
# True ATE = 1.5
df_cont['Y_cont'] = 1.5*df_cont['T_cont'] + 0.1*df_cont['age'] + np.random.normal(0, 1, n)

# --- EXECUTION ---

print("--- RCT Methods ---")
print(f"True ATE: 2.0")

# Estimate ATE and Confidence Interval
ate, lower, upper = calculate_ate_ci(df)
print(f"ATE (CI): {ate:.3f} [{lower:.3f}, {upper:.3f}]")

# Estimate ATE and P-Value
ate, t_stat, p_val = calculate_ate_pvalue(df)
print(f"ATE (p-val): {ate:.3f} (p={p_val:.3f})")

print("\n--- Propensity Score Methods ---")
print(f"True ATE: 2.0")

# Define the formula for the propensity score model (T ~ age + income)
ps_formula = "age + income"

# Inverse Propensity Weighting
ate_ipw = ipw(df, ps_formula, 'T', 'Y')
print(f"IPW ATE: {ate_ipw:.3f}")

# Doubly Robust Estimator (IPW + Outcome Regression)
ate_dr = doubly_robust(df, ps_formula, 'T', 'Y')
print(f"Doubly Robust ATE: {ate_dr:.3f}")

print("\n--- Meta-Learners ---")

# Split data for training and inference
train = df.iloc[:800].copy()
test = df.iloc[800:].copy()
covariates = ['age', 'income']

# A. Discrete Learners (Binary Treatment)
# ---------------------------------------
print("A. Discrete Learners (True ATE: 2.0)")

# S-Learner (Single Model)
res_s = s_learner_discrete(train, test, covariates, 'T', 'Y')
print(f"S-Learner CATE (head): \n{res_s['cate'].head(3).values}")
print(f"S-Learner Average CATE: {res_s['cate'].mean():.3f}")
print("")

# T-Learner (Two Models)
res_t = t_learner_discrete(train, test, covariates, 'T', 'Y')
print(f"T-Learner CATE (head): \n{res_t['cate'].head(3).values}")
print(f"T-Learner Average CATE: {res_t['cate'].mean():.3f}")
print("")

# X-Learner (Cross Learners)
res_x = x_learner_discrete(train, test, covariates, 'T', 'Y')
print(f"X-Learner CATE (head): \n{res_x['cate'].head(3).values}")
print(f"X-Learner Average CATE: {res_x['cate'].mean():.3f}")
print("")

# B. Continuous Learner (Continuous Treatment)
# --------------------------------------------
print("\nB. Continuous Learner (True ATE: 1.5)")
train_c = df_cont.iloc[:800].copy()
test_c = df_cont.iloc[800:].copy()

# Double ML (Residualization)
res_dml = double_ml_cate(train_c, test_c, covariates, 'T_cont', 'Y_cont')
print(f"Double ML CATE (head): \n{res_dml['cate'].head(3).values}")
print(f"Double ML Average CATE: {res_dml['cate'].mean():.3f}")
```

---

## 📚 API Reference

### 1. Randomized Controlled Trials (`causal_toolkit_tc.rct`)
Functions for analyzing standard A/B tests.

- **`calculate_ate_ci(data)`**
  - Calculates the Average Treatment Effect (ATE) and 95% Confidence Intervals.
  - **Input:** DataFrame with columns `T` (treatment) and `Y` (outcome).
  - **Returns:** `(ate, ci_lower, ci_upper)`

- **`calculate_ate_pvalue(data)`**
  - Calculates the ATE along with the t-statistic and p-value for hypothesis testing.
  - **Input:** DataFrame with columns `T` (treatment) and `Y` (outcome).
  - **Returns:** `(ate, t_stat, p_value)`

### 2. Propensity Score Methods (`causal_toolkit_tc.propensity`)
Methods for observational data to adjust for confounding variables.

- **`ipw(df, ps_formula, T, Y)`**
  - Estimates ATE using **Inverse Propensity Weighting**.
  - **Input:** `df` (data), `ps_formula` (e.g., "age + income"), `T` (col name), `Y` (col name).
  - **Returns:** Estimated ATE (float).

- **`doubly_robust(df, ps_formula, T, Y)`**
  - Estimates ATE using a **Doubly Robust** estimator (combines IPW with a regression model). More stable than IPW alone.
  - **Input:** `df` (data), `ps_formula` (e.g., "age + income"), `T` (col name), `Y` (col name).
  - **Returns:** Estimated ATE (float).

### 3. Meta-Learners (`causal_toolkit_tc.meta_learners`)
Machine learning methods for estimating Heterogeneous Treatment Effects (CATE).

- **`s_learner_discrete(train, test, X, T, y)`**
  - **S-Learner (Single):** Uses a single regression model with treatment as a feature.
  
- **`t_learner_discrete(train, test, X, T, y)`**
  - **T-Learner (Two):** Fits two separate models (one for treated units, one for control).

- **`x_learner_discrete(train, test, X, T, y)`**
  - **X-Learner:** A multi-stage learner best suited for unbalanced treatment groups (e.g., when the control group is much larger).

- **`double_ml_cate(train, test, X, T, y)`**
  - **Double ML:** Uses residualization (two-stage regression) to isolate the treatment effect. Best for continuous treatments or complex confounding.
