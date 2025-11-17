# %% [markdown]
# Assignment 3
# 
# Overview
# This coding assignment is based on the lecture and notebook of Module 3 - ATE Estimation: Linear Regression, Propensity Score, Doubly Robust Estimator.  
# 
# Students will implement functions to estimate the Average Treatment Effect (ATE) using the IPW and doubly robust estimators.
# 
# 
# Data Format  
# Students will work with a pandas DataFrame containing three columns:  
# 
# Column	Description  
# X	Covariates: One or more confounding variables  
# T	Treatment assignment (1 = treatment, 0 = control)  
# Y	Outcome (the variable being measured)  

# %%


# %%
# Generate df_test_1

# Here is the code to generate two of the sample data sets that are going to be used by Gradescope:  
import numpy as np
import pandas as pd

# Test 1: IPW with simple positive treatment effect
np.random.seed(42)
n = 1000

# Generate data with known ATE = 2
x = np.random.normal(0, 1, n)
prob_t = 1 / (1 + np.exp(-(0.5 * x)))
t = np.random.binomial(1, prob_t, n)
y = 2 * t + x + np.random.normal(0, 0.5, n)

df_test_1 = pd.DataFrame({'x': x, 't': t, 'y': y})

# Review the generated dataframe
print("df_test_1:")
df_test_1.head()

# %%
# Generate df_test_5

# Test 5: IPW with categorical covariate
np.random.seed(101)
n = 1000

# Generate data with categorical confounder
group = np.random.choice(['A', 'B', 'C'], n)
group_effect = {'A': 0, 'B': 1, 'C': 2}
x_numeric = np.array([group_effect[g] for g in group])

prob_t = 1 / (1 + np.exp(-(0.5 * x_numeric)))
t = np.random.binomial(1, prob_t, n)
y = 2.0 * t + x_numeric + np.random.normal(0, 0.5, n)

df_test_5 = pd.DataFrame({'group': group, 't': t, 'y': y})

# Review the generated dataframe
print("\ndf_test_5:")
df_test_5.head()

# %%
"""
Task 1: Implement ipw(df, ps_formula, T, Y)
Calculate the Average Treatment Effect using Inverse Propensity Weighting.

Function Signature:

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
Parameters:

df: DataFrame containing treatment, outcome, and covariates
ps_formula: Patsy formula string for propensity score model (e.g., "age + income")
T: Name of the treatment variable (string)
Y: Name of the outcome variable (string)
Returns:

float: The estimated ATE
Key Concepts:

The IPW estimator uses propensity scores to create pseudo-populations where treatment assignment is independent of confounders.

Formula:

ATE = E[(T - e(X)) / (e(X) × (1 - e(X))) × Y]
where e(X) = P(T=1|X) is the propensity score.

Implementation Steps:

Convert formula to design matrix using patsy.dmatrix(ps_formula, df)
Fit propensity score model using LogisticRegression(penalty=None, max_iter=1000)
Get propensity scores: ps = model.predict_proba(X)[:, 1]
Calculate ATE using the IPW formula
"""

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    from patsy import dmatrix
    from sklearn.linear_model import LogisticRegression

    # 1. Convert formula to design matrix using patsy.dmatrix(ps_formula, df)
    X = dmatrix(ps_formula, df)

    # 2. Fit propensity score model using LogisticRegression(penalty=None, max_iter=1000)
    ps_model = LogisticRegression(penalty=None, max_iter=1000)

    # Convert treatment and outcome to arrays
    T_series = df[T].astype(int).to_numpy()
    Y_series = df[Y].astype(float).to_numpy()


    # Fit the model
    ps_model.fit(X, T_series)

    # 3. Get propensity scores: ps = model.predict_proba(X)[:, 1]
    ps = ps_model.predict_proba(X)[:, 1]

    # Calibrate propensity scores

    # 4. Calculate ATE using the IPW formula
    ate = ((T_series * Y_series / ps) - ((1 - T_series) * Y_series / (1 - ps))).mean()

    return ate

# Example use
df = pd.DataFrame({
    'treatment': [0, 1, 0, 1, 1],
    'outcome': [1.0, 2.5, 1.2, 2.8, 3.0],
    'age': [25, 30, 35, 40, 28]
})

# Calculate IPW ATE using df_test_1
ate = ipw(df = df_test_1, ps_formula = "x", T = "t", Y = "y")
print(f"df_test_1 Estimated ATE : {ate}")

# Calculate IPW ATE using df_test_5
ate = ipw(df = df_test_5, ps_formula = "group", T = "t", Y = "y")
print(f"df_test_5 Estimated ATE : {ate}") 

# %%


# %%
"""
Task 2: Implement doubly_robust(df, formula, T, Y)
Calculate the Average Treatment Effect using Doubly Robust Estimation.

Function Signature:

def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
Parameters:

df: DataFrame containing treatment, outcome, and covariates
formula: Patsy formula string for both models (e.g., "age + income")
T: Name of the treatment variable (string)
Y: Name of the outcome variable (string)
Returns:

float: The estimated ATE
Key Concepts:

The doubly robust estimator combines propensity scores and outcome regression models. It's called "doubly robust" because it gives consistent estimates if either the propensity score model or the outcome models are correctly specified (but not necessarily both).

Formula:

ATE = E[T(Y - μ₁(X))/e(X) + μ₁(X)] - E[(1-T)(Y - μ₀(X))/(1-e(X)) + μ₀(X)]
where:

e(X) = P(T=1|X) (propensity score)
μ₁(X) = E[Y|T=1,X] (outcome model for treated)
μ₀(X) = E[Y|T=0,X] (outcome model for control)
Implementation Steps:

1. Convert formula to design matrix using patsy.dmatrix(formula, df)
2. Fit propensity score model: LogisticRegression(penalty=None, max_iter=1000)
3. Fit outcome model for control group: LinearRegression().fit(X[T==0], Y[T==0])
4. Fit outcome model for treated group: LinearRegression().fit(X[T==1], Y[T==1])
5. Get predictions for all observations from both outcome models
6. Calculate ATE using the DR formula
"""


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """
    Parameters:

    df: DataFrame containing treatment, outcome, and covariates
    formula: Patsy formula string for both models (e.g., "age + income")
    T: Name of the treatment variable (string)
    Y: Name of the outcome variable (string)
    Returns:

    float: The estimated ATE
    """

    from patsy import dmatrix
    from sklearn.linear_model import LogisticRegression

    # 1. Convert formula to design matrix using patsy.dmatrix(formula, df)
    X = dmatrix(formula, df)

    # 2. Fit propensity score model: LogisticRegression(penalty=None, max_iter=1000)
    ps_model = LogisticRegression(penalty=None, max_iter=1000) # Initialize model
    T_series = df[T].astype(int) # Convert treatment to int
    Y_series = df[Y].astype(float) # Convert outcome to float
    ps_model.fit(X, T_series) # Fit model
    ps = pd.Series(ps_model.predict_proba(X)[:, 1]) # Get propensity scores

    # 3. Fit outcome model for control group: LinearRegression().fit(X[T==0], Y[T==0])
    from sklearn.linear_model import LinearRegression
    outcome_model_control = LinearRegression() # Initialize model
    outcome_model_control.fit(X[T_series==0], Y_series[T_series==0]) # Fit model

    outcome_model_treated = LinearRegression() # Initialize model
    outcome_model_treated.fit(X[T_series==1], Y_series[T_series==1]) # Fit model

    # 5. Get predictions for all observations from both outcome models
    mu_0 = pd.Series(outcome_model_control.predict(X))  # Predictions for control
    mu_1 = pd.Series(outcome_model_treated.predict(X))  # Predictions for treated

    # 6. Calculate ATE using the DR formula
    dr_ate = np.mean(mu_1 - mu_0 + (T_series * (Y_series - mu_1) / ps) - ((1 - T_series) * (Y_series - mu_0) / (1 - ps)))
    return(dr_ate)

# Calculate doubly robust ATE using df_test_1
ate = doubly_robust(df = df_test_1, formula = "x", T = "t", Y = "y")
print(f"df_test_1 Doubly Robust Estimated ATE : {ate}")

# Calculate doubly robust ATE using df_test_5
ate = doubly_robust(df = df_test_5, formula = "group", T = "t", Y = "y")
print(f"df_test_5 Doubly Robust Estimated ATE : {ate}")