# Imports
import numpy as np
import pandas as pd

from causal_toolkit_tc import (
    s_learner_discrete,
    t_learner_discrete,
    x_learner_discrete,
    double_ml_cate
)

# Data Generation Functions (add to the top of test_meta_learners_week05.py)
def simple_data():
    """Generate simple data with known treatment effect"""
    import numpy as np
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

def continuous_treatment_data():
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

def test_s_learner_returns_dataframe():
    # 1. Generate test data 
    train, test = simple_data() 

    # 2. Call function under test 
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y') 

    # 3. Assert expected behavior 
    assert isinstance(result, pd.DataFrame)

def test_s_learner_has_cate_column():	
    #Has 'cate' column
    train, test = simple_data() # Generate test data
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y') # Call function under test
    assert 'cate' in result.columns # Assert expected behavior

def test_s_learner_has_cate_column():	
    #Has 'cate' column
    train, test = simple_data() # Generate test data
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y') # Call function under test
    assert 'cate' in result.columns # Assert expected behavior

def test_s_learner_constant_effect():	
    # Recovers the true effect
    train, test = simple_data() # Generate test data
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y') # Call function under test
    estimated_effect = result['cate'].mean() # Calculate estimated effect
    assert abs(estimated_effect - 2.0) < 0.5 # Assert expected behavior

def test_s_learner_return_numeric_cate():
    # Numeric CATE values
    train, test = simple_data() # Generate test data
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y') # Call function under test
    assert pd.api.types.is_numeric_dtype(result['cate']) # Assert expected behavior

def test_s_learner_no_nan_values():	
    # No NaN in CATE
    train, test = simple_data() # Generate test data
    result = s_learner_discrete(train, test, ['x1', 'x2'], 't', 'y') # Call function under test
    assert not result['cate'].isnull().any() # Assert expected behavior

def test_t_learner_returns_dataframe():	
    # Returns DataFrame
    train, test = simple_data()
    result = t_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)

def test_x_learner_returns_dataframe():	
    # Returns DataFrame
    train, test = simple_data()
    result = x_learner_discrete(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)

def test_double_ml_returns_dataframe():	
    # Returns DataFrame
    train, test = simple_data()
    result = double_ml_cate(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)

def test_double_ml_continuous_treatment():	
    # Works with continuous T
    train, test = continuous_treatment_data()
    result = double_ml_cate(train, test, ['x1', 'x2'], 't', 'y')
    assert isinstance(result, pd.DataFrame)