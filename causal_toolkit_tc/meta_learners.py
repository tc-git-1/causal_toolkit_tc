"""
Assignment 4

Overview
This coding assignment is based on the lecture and notebook of Module 4 - 
CATE Estimation: Transformed Outcome, Meta-learners, Double ML. 

In this assignment, you will implement four meta-learning approaches for estimating heterogeneous treatment effects:
1. S-Learner - For discrete treatment
2. T-Learner - For discrete treatment
3. X-Learner - For discrete treatment
4. Double ML - For continuous treatment

Data Format
Students will work with a pandas DataFrame containing three columns:

Column	Description
X	Covariates: One or more confounding variables
T	Treatment assignment (1 = treatment, 0 = control)
Y	Outcome (the variable being measured)

The DataFrame will be split into a training DataFrame and a test DataFrame.

Here is the code to generate two of the sample data sets that are going to be used by Gradescope:
"""

# Imports
import numpy as np
import pandas as pd
from typing import List
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor # # Use LGBMRegressor() for all regression models (default parameters are fine)

class DataGenerator:
    def simple_data(self):
        """Generate simple data with known treatment effect"""
        np.random.seed(42)
        n = 1000
        
        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Treatment assignment (confounded)
        prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
        t = np.random.binomial(1, prob_t, n)
        
        # Outcome with constant treatment effect = 2.0
        y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
        
        # Split into train/test
        train = df.iloc[:800].copy()
        test = df.iloc[800:].copy()
        
        return train, test

    def heterogeneous_data(self):
        """Generate data with heterogeneous treatment effect"""
        np.random.seed(123)
        n = 1500

        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # Treatment assignment
        prob_t = 1 / (1 + np.exp(-(0.4 * x1)))
        t = np.random.binomial(1, prob_t, n)

        # Outcome with heterogeneous effect: effect depends on x1
        # CATE(x1) = 1 + 0.5*x1
        te = 1.0 + 0.5 * x1
        y = te * t + x1 + 0.3 * x2 + np.random.normal(0, 0.5, n)

        df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

        train = df.iloc[:1200].copy()
        test = df.iloc[1200:].copy()

        return train, test

    def continuous_treatment_data(self):
        """Generate data with continuous treatment"""
        np.random.seed(789)
        n = 1000

        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # Continuous treatment
        t = 10 + x1 + 2*x2 + np.random.normal(0, 1, n)

        # Outcome: linear effect of treatment
        y = t + x1 + 0.5*x2 + np.random.normal(0, 0.5, n)

        df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

        train = df.iloc[:800].copy()
        test = df.iloc[800:].copy()

        return train, test

# General Tips
# 1. Use LGBMRegressor() for all regression models (default parameters are fine)
# 2. Use LogisticRegression(penalty=None) for propensity score estimation
# 3. Return test DataFrame with added 'cate' column


"""
Task 1: Implement s_learner_discrete(train, test, X, T, y)

Function Signature:

def s_learner_discrete(train, test, X, T, y) -> pd.DataFrame:

Parameters:
train: a training DataFrame containing treatment, outcome, and covariates
test: a test DataFrame containing treatment, outcome, and covariates
X: a list of the names of the covariates (e.g., ["age", "income"])
T: Name of the treatment variable (string). The treatment variable will be binary.
y: Name of the outcome variable (string)

Returns:
pd.DataFrame: a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame.

Key Concepts:
Fit a single model μ(X, T) that predicts Y from both covariates X and treatment T
Estimate CATE by comparing predictions: CATE(x) = μ(x, T=1) - μ(x, T=0)
"""

def s_learner_discrete(train: pd.DataFrame,
                       test: pd.DataFrame,
                       X: List[str],
                       T: str,
                       y: str) -> pd.DataFrame:
    """
    S-Learner for binary treatment.
    
    Trains a single outcome model mu(X, T) using LGBMRegressor on the training set,
    then computes CATE(x) on the test set by:
        CATE(x) = mu(x, T=1) - mu(x, T=0)

    Parameters
    ----------
    train : pd.DataFrame
        Training data containing covariates X, treatment T (0/1), and outcome y.
    test : pd.DataFrame
        Test data containing covariates X, treatment T (0/1), and outcome y.
    X : list of str
        Names of covariate columns.
    T : str
        Name of the (binary) treatment column.
    y : str
        Name of the outcome column.

    Returns
    -------
    pd.DataFrame
        A **copy** of the test DataFrame with an added column 'cate' holding the
        estimated conditional average treatment effect for each row.
    """
    # Defensive copy of feature list (avoid duplicating T if user accidentally included it in X)
    X = [c for c in X if c != T]

    # Basic column checks
    missing_train = [c for c in (X + [T, y]) if c not in train.columns]
    missing_test  = [c for c in (X + [T, y]) if c not in test.columns]
    
    if missing_train:
        raise ValueError(f"Train is missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test is missing columns: {missing_test}")

    # ---- Fit single outcome model mu(X, T) ----
    model = LGBMRegressor(random_state=123, verbosity=-1)
    model.fit(train[X + [T]], train[y])

    # ---- Predict counterfactual outcomes on test: T=1 and T=0 ----
    X_base = test[X].copy()

    X1 = X_base.copy() 
    X1[T] = 1 # set treatment to 1
    mu1 = model.predict(X1[X + [T]]) # Predict outcome if treated

    X0 = X_base.copy()
    X0[T] = 0 # set treatment to 0
    mu0 = model.predict(X0[X + [T]]) # Predict outcome if control

    # ---- CATE = mu(x,1) - mu(x,0) ----
    cate = mu1 - mu0 # CATE estimate, outcome if treated minus outcome if control

    out = test.copy() # Create a copy of test DataFrame
    out["cate"] = cate # Add CATE column
    return out # Return the modified test DataFrame


"""
Task 2: Implement t_learner_discrete(train, test, X, T, y)
Function Signature:
def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:

Parameters:
train: a training DataFrame containing treatment, outcome, and covariates
test: a test DataFrame containing treatment, outcome, and covariates
X: a list of the names of the covariates (e.g., ["age", "income"])
T: Name of the treatment variable (string). The treatment variable will be binary.
y: Name of the outcome variable (string)

Returns:
pd.DataFrame: a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame.

Key Concepts:
Fit two separate models:
μ₀(X) on control group (T=0)
μ₁(X) on treated group (T=1)
Estimate CATE: CATE(x) = μ₁(x) - μ₀(x)
"""

def t_learner_discrete(train: pd.DataFrame,
                       test: pd.DataFrame,
                       X: List[str],
                       T: str,
                       y: str) -> pd.DataFrame:
    """
    T-Learner for binary treatment.
    
    Trains two separate outcome models mu0(X) and mu1(X) using LGBMRegressor on the training set,
    then computes CATE(x) on the test set by:
        CATE(x) = mu1(x) - mu0(x)
    Parameters
    ----------
    train : pd.DataFrame
        Training data containing covariates X, treatment T (0/1), and outcome y.
    test : pd.DataFrame
        Test data containing covariates X, treatment T (0/1), and outcome y.
    X : list of str
        Names of covariate columns.
    T : str
        Name of the (binary) treatment column.
    y : str
        Name of the outcome column.     
    Returns
    -------
    pd.DataFrame
        A **copy** of the test DataFrame with an added column 'cate' holding the
         estimated conditional average treatment effect for each row.
    """
    # Defensive copy of feature list (avoid duplicating T if user accidentally included it in X)
    X = [c for c in X if c != T]

    # Basic column checks
    missing_train = [c for c in (X + [T, y]) if c not in train.columns]
    missing_test  = [c for c in (X + [T, y]) if c not in test.columns]
    if missing_train:
        raise ValueError(f"Train is missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test is missing columns: {missing_test}")

    # Split training data into treated and control groups
    train_control = train[train[T] == 0]
    train_treated = train[train[T] == 1]

    # Fit outcome models for control
    mu0_model = LGBMRegressor()
    mu0_model.fit(train_control[X], train_control[y])

    # Fit outcome models for treated
    mu1_model = LGBMRegressor()
    mu1_model.fit(train_treated[X], train_treated[y])

    # Predict potential outcomes on test set
    mu0_pred = mu0_model.predict(test[X]) # Predict outcome if control
    mu1_pred = mu1_model.predict(test[X]) # Predict outcome if treated

    # Compute CATE
    test_copy = test.copy()
    test_copy['cate'] = mu1_pred - mu0_pred # CATE estimate, outcome if treated minus outcome if control

    return test_copy


"""
Task 3: Implement x_learner_discrete(train, test, X, T, y)

Function Signature:
def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:

Parameters:
train: a training DataFrame containing treatment, outcome, and covariates
test: a test DataFrame containing treatment, outcome, and covariates
X: a list of the names of the covariates (e.g., ["age", "income"])
T: Name of the treatment variable (string). The treatment variable will be binary.
y: Name of the outcome variable (string)

Returns:
pd.DataFrame: a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame.
Key Concepts:

Stage 1: Fit outcome models μ₀ and μ₁ like T-Learner

Stage 2: Compute pseudo-treatment effects:
τ̂₀(xᵢ) = μ₁(xᵢ) - yᵢ for control units
τ̂₁(xᵢ) = yᵢ - μ₀(xᵢ) for treated units
Fit models for τ₀(x) and τ₁(x)

Final estimate: CATE(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x) where e(x) is the propensity score
"""

import numpy as np
import pandas as pd
from typing import List
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

def x_learner_discrete(train: pd.DataFrame,
                       test: pd.DataFrame,
                       X: List[str],
                       T: str,
                       y: str,
                       reweight_first_stage: bool = True) -> pd.DataFrame:
    """
    X-Learner for binary treatment with optional inverse-propensity reweighting in Stage 1.
    Returns a copy of `test` with a 'cate' column.
    """
    # ---- Defensive features & validation ----
    X = [c for c in X if c not in (T, y)]
    
    need = set(X + [T, y])
    for name, df in [('train', train), ('test', test)]:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    if train[T].nunique() < 2:
        raise ValueError("Training set must contain both treatment arms (0 and 1).")

    # Split arms
    t0 = train[train[T] == 0].copy()
    t1 = train[train[T] == 1].copy()

    # ---- Propensity model e(x) = P(T=1|X) ----
    try:
        ps = LogisticRegression()
    except Exception:
        # Some sklearn versions don't accept 'none'
        ps = LogisticRegression(max_iter=1000)
    ps.fit(train[X], train[T])

    # ---- Stage 1: outcome models m0 and m1 ----
    lgbm_params = dict(random_state=123, force_col_wise=True, verbosity=-1)
    m0 = LGBMRegressor(**lgbm_params)  # E[Y|X, T=0]
    m1 = LGBMRegressor(**lgbm_params)  # E[Y|X, T=1]

    if reweight_first_stage:
        e0 = np.clip(ps.predict_proba(t0[X])[:, 1], 1e-6, 1-1e-6)
        e1 = np.clip(ps.predict_proba(t1[X])[:, 1], 1e-6, 1-1e-6)
        w0 = 1.0 / (1.0 - e0)   # controls
        w1 = 1.0 / e1           # treated
        m0.fit(t0[X], t0[y], sample_weight=w0)
        m1.fit(t1[X], t1[y], sample_weight=w1)
    else:
        m0.fit(t0[X], t0[y])
        m1.fit(t1[X], t1[y])

    # ---- Stage 2: pseudo-effects (correct signs!) ----
    # controls: tau_hat0 = m1(x_control) - y_control
    tau_hat0 = m1.predict(t0[X]) - t0[y].to_numpy()
    # treated:  tau_hat1 = y_treated      - m0(x_treated)
    tau_hat1 = t1[y].to_numpy() - m0.predict(t1[X])

    # Effect models on each arm
    m_tau0 = LGBMRegressor(**lgbm_params)  # trained on controls
    m_tau1 = LGBMRegressor(**lgbm_params)  # trained on treated
    m_tau0.fit(t0[X], tau_hat0)
    m_tau1.fit(t1[X], tau_hat1)

    # ---- Stage 3: blend with propensity (textbook) ----
    e_test = ps.predict_proba(test[X])[:, 1]
    tau0_pred = m_tau0.predict(test[X])  # effect on control pop
    tau1_pred = m_tau1.predict(test[X])  # effect on treated pop
    cate = e_test * tau0_pred + (1.0 - e_test) * tau1_pred

    out = test.copy()
    out["cate"] = cate
    return out


"""
Task 4: Implement double_ml_cate(train, test, X, T, y)
Function Signature:

def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:

Parameters:

train: a training DataFrame containing treatment, outcome, and covariates
test: a test DataFrame containing treatment, outcome, and covariates
X: a list of the names of the covariates (e.g., ["age", "income"])
T: Name of the treatment variable (string). The treatment variable will be continuous.
y: Name of the outcome variable (string)
Returns:

pd.DataFrame: a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame.
Key Concepts:

Partial out X from both T and Y using cross-fitting:
T_res = T - Ê[T|X]
Y_res = Y - Ê[Y|X]
Create transformed outcome: Y* = Y_res / T_res
Create weights: w = T_res²
Fit CATE model: τ̂(x) on X using Y* as outcome and w as sample weights
"""

def double_ml_cate(train: pd.DataFrame,
                   test: pd.DataFrame,
                   X: List[str],
                   T: str,
                   y: str) -> pd.DataFrame:
    """
    Double Machine Learning for continuous treatment.
    
    Implements the Double ML algorithm to estimate CATE(x) on the test set.
    
    Parameters
    ----------
    train : pd.DataFrame
        Training data containing covariates X, continuous treatment T, and outcome y.
    test : pd.DataFrame
        Test data containing covariates X, continuous treatment T, and outcome y.
    X : list of str
        Names of covariate columns.
    T : str
        Name of the (continuous) treatment column.
    y : str
        Name of the outcome column.     
    Returns
    -------
    pd.DataFrame
        A **copy** of the test DataFrame with an added column 'cate' holding the
         estimated conditional average treatment effect for each row.
    """
    # Defensive copy of feature list (avoid duplicating T if user accidentally included it in X)
    X = [c for c in X if c != T]
    
    # Basic column checks
    missing_train = [c for c in (X + [T, y]) if c not in train.columns]
    missing_test  = [c for c in (X + [T, y]) if c not in test.columns]
    if missing_train:
        raise ValueError(f"Train is missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test is missing columns: {missing_test}")
    
    # ---- Step 1: Partial out X from T and Y ----
    
    # Model for T
    t_model = LGBMRegressor()
    t_model.fit(train[X], train[T])
    T_hat = t_model.predict(train[X])
    T_res = train[T] - T_hat
    # Model for Y
    y_model = LGBMRegressor()
    y_model.fit(train[X], train[y])
    Y_hat = y_model.predict(train[X])
    Y_res = train[y] - Y_hat
    # ---- Step 2: Create transformed outcome and weights ----
    Y_star = Y_res / T_res
    weights = T_res ** 2
    # ---- Step 3: Fit CATE model tau(x) ----
    cate_model = LGBMRegressor()
    cate_model.fit(train[X], Y_star, sample_weight=weights)
    # ---- Step 4: Predict CATE on test set ----
    cate_pred = cate_model.predict(test[X])
    test_copy = test.copy()
    test_copy['cate'] = cate_pred
    return test_copy