# causal_toolkit_tc

A Python package for causal inference methods including ATE estimation, propensity score methods, and meta-learners.

---

## 📦 Features
- **RCT Methods**: `calculate_ate_ci()`, `calculate_ate_pvalue()`
- **Propensity Score**: `ipw()`, `doubly_robust()`
- **Meta-Learners**: `s_learner_discrete()`, `t_learner_discrete()`, `x_learner_discrete()`, `double_ml_cate()`

---

## ✅ Installation
```bash
git clone https://github.com/yourusername/causal_toolkit_tc.git
cd causal_toolkit_tc
uv pip install -e .[dev]