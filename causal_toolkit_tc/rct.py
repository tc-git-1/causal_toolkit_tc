# %% [markdown]
# Overview
# This coding assignment is based on the lecture and notebook of Module 2 -   
# ATE Estimation: Randomized Experiments, Stats Review.   
# Students will implement functions to perform statistical inference on the Average Treatment Effect (ATE) from Randomized Controlled Trial (RCT) data.  
#  
# 
# Data Format  
# Students will work with a pandas DataFrame containing three columns:  
# 
# Column	Description  
# I	User ID (identifier)  
# T	Treatment assignment (1 = treatment, 0 = control)  
# Y	Outcome (the variable being measured)

# %%
# Imports
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

# Create a class to encapsulate data generation
class DataGenerator:
    @classmethod
    def generate_data(cls):
        # Positive effect data
        np.random.seed(42)
        n = 1000
        cls.positive_effect_data = pd.DataFrame({
            'I': range(n),
            'T': np.random.binomial(1, 0.5, n),
        })
        cls.positive_effect_data['Y'] = np.where(
            cls.positive_effect_data['T'] == 1, 
            np.random.normal(10, 2, n),
            np.random.normal(8, 2, n)
        )

        # No effect data
        np.random.seed(123)
        n = 500
        cls.no_effect_data = pd.DataFrame({
            'I': range(n),
            'T': np.random.binomial(1, 0.5, n),
        })
        cls.no_effect_data['Y'] = np.random.normal(5, 3, n)

        return cls.positive_effect_data, cls.no_effect_data

# Generate the data
positive_data, no_effect_data = DataGenerator.generate_data()

print("Positive Effect Data:")
print(positive_data)
print()

print("No Effect Data:")
print(no_effect_data)

# %% [markdown]
# Tasks
# Implement two functions, calculate_ate_ci() and calculate_ate_pvalue(), as listed below.
# 
# Save it as week02.py and submit to Gradescope.
#  
# ## Task 1: Implement calculate_ate_ci()
# Calculate the confidence interval for the Average Treatment Effect.
# 
# Function Signature:
# 
# def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
#  
# 
# Returns:
# 
# ATE_estimate: The estimated average treatment effect
# CI_lower: Lower bound of the confidence interval
# CI_upper: Upper bound of the confidence interval
# Key Concepts:
# 
# ATE = E[Y|T=1] - E[Y|T=0]
# Standard Error: SE(ATE) = sqrt(Var(Y|T=1)/n₁ + Var(Y|T=0)/n₀)
# Confidence Interval: ATE ± z_(α/2) × SE(ATE)

# %%
# Task 1: Implement calculate_ate_ci()
def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Calculate the Average Treatment Effect (ATE) and its confidence interval.
    
    Parameters:
    - data: DataFrame with columns 'I', 'T', and 'Y'
    I User ID (identifier)
    T Treatment assignment (1 = treatment, 0 = control)
	Y Outcome (the variable being measured)
    
    - alpha: Significance level for the confidence interval
    
    Returns:
    - ATE, lower bound of CI, upper bound of CI
    """
    # ATE = E[Y|T=1] - E[Y|T=0]
    Y_treated = data[data['T'] == 1]['Y']
    Y_control = data[data['T'] == 0]['Y']
    ate = Y_treated.mean() - Y_control.mean()
    
    # Standard Error: SE(ATE) = sqrt(Var(Y|T=1)/n₁ + Var(Y|T=0)/n₀)
    var_treated = Y_treated.var(ddof=1) # sample variance is used (this is standard for estimating standard errors)
    var_control = Y_control.var(ddof=1) # sample variance is used (this is standard for estimating standard errors)
    n_treated = len(Y_treated)
    n_control = len(Y_control)
    se_ate = np.sqrt(var_treated / n_treated + var_control / n_control)   
    
    # Confidence interval calculation
    z_score = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - (z_score * se_ate)
    ci_upper = ate + (z_score * se_ate)
    
    return ate, ci_lower, ci_upper

# Example use
ate, ci_lower, ci_upper = calculate_ate_ci(positive_data)
print(f"Positive Data ATE:  {round(ate, 2)}, 95% CI: ({round(ci_lower, 2)}, {round(ci_upper, 2)})")

# Example use
ate, ci_lower, ci_upper = calculate_ate_ci(no_effect_data)
print(f"No Effect Data ATE: {round(ate, 2)}, 95% CI: ({round(ci_lower, 2)}, {round(ci_upper, 2)})")

# %% [markdown]
# ## Task 2: Implement calculate_ate_pvalue()  
# Calculate the p-value for testing whether the ATE differs from a null hypothesis value.  
# 
# Function Signature:  
# def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:  
# 
# Returns:  
# 
# ATE_estimate: The estimated average treatment effect  
# t_statistic: The test statistic  
# p_value: The two-sided p-value  
# Key Concepts:  
# 
# Null Hypothesis: H₀: ATE = 0  
# Alternative Hypothesis: H₁: ATE ≠ 0  
# Test Statistic: t = (ATE_estimate - 0) / SE(ATE)  
# P-value: 2 × (1 - Φ(|t|)) where Φ is the standard normal CDF  

# %%
# Task 2: Implement calculate_ate_pvalue()  
def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculate the Average Treatment Effect (ATE) and its p-value using a two-sample t-test.
    
    Parameters:
    - data: DataFrame with columns 'I', 'T', and 'Y'
    I User ID (identifier)
    T Treatment assignment (1 = treatment, 0 = control)
    Y Outcome (the variable being measured)
    
    Returns:
    - ATE, t-statistic, p-value
    """
    ate = calculate_ate_ci(data)[0]
    Y_treated = data[data['T'] == 1]['Y']
    Y_control = data[data['T'] == 0]['Y']
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(Y_treated, Y_control, equal_var=False) # Welch's t-test
    
    return ate, t_stat, p_value

# Example use
ate, t_stat, p_value = calculate_ate_pvalue(positive_data)
print(f"Positive Data ATE:  {round(ate, 2)}, t-statistic: {round(t_stat, 2)}, p-value: {round(p_value, 4)}") 

# Example use
ate, t_stat, p_value = calculate_ate_pvalue(no_effect_data)
print(f"No Effect Data ATE: {round(ate, 2)}, t-statistic: {round(t_stat, 2)}, p-value: {round(p_value, 4)}")