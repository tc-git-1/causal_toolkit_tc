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
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:

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

    # 1. Convert formula to design matrix using patsy.dmatrix(formula, df)
    X = dmatrix(formula, df)

    # 2. Fit propensity score model: LogisticRegression(penalty=None, max_iter=1000)
    ps_model = LogisticRegression(penalty=None, max_iter=1000) # Initialize model
    T_series = df[T].astype(int) # Convert treatment to int
    Y_series = df[Y].astype(float) # Convert outcome to float
    ps_model.fit(X, T_series) # Fit model
    ps = pd.Series(ps_model.predict_proba(X)[:, 1]) # Get propensity scores

    # 3. Fit outcome model for control group: LinearRegression().fit(X[T==0], Y[T==0])
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