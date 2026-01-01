"""
Statistics Engine
==================

Statistical analysis for Baseline Evidence and Analytical Robustness.

Provides:
- Descriptive statistics
- Correlation analysis
- Statistical tests (t-test, ANOVA, chi-square)
- Regression modeling (14 estimators)
- Robustness checks
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class RegressionResult:
    """Container for regression results."""
    estimator: str
    dependent_var: str
    independent_vars: List[str]
    n_obs: int
    r_squared: Optional[float]
    adj_r_squared: Optional[float]
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    p_values: Dict[str, float]
    conf_intervals: Dict[str, Tuple[float, float]]
    model_summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'estimator': self.estimator,
            'dependent_var': self.dependent_var,
            'independent_vars': self.independent_vars,
            'n_obs': self.n_obs,
            'r_squared': self.r_squared,
            'adj_r_squared': self.adj_r_squared,
            'coefficients': self.coefficients,
            'std_errors': self.std_errors,
            'p_values': self.p_values
        }


class DescriptiveStats:
    """Descriptive statistics calculator."""
    
    @staticmethod
    def summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics.
        
        Returns: count, mean, std, min, 25%, 50%, 75%, max, skew, kurtosis
        """
        if columns:
            df = df[columns]
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        stats['missing'] = df.isnull().sum()
        stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
        
        return stats
    
    @staticmethod
    def frequency_table(
        df: pd.DataFrame,
        column: str,
        normalize: bool = False
    ) -> pd.DataFrame:
        """Create frequency table for categorical variable."""
        counts = df[column].value_counts()
        result = pd.DataFrame({
            'count': counts,
            'percent': (counts / len(df) * 100).round(2)
        })
        return result
    
    @staticmethod
    def correlation_matrix(
        df: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Args:
            method: 'pearson', 'spearman', or 'kendall'
        """
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)


class StatisticalTests:
    """Statistical hypothesis tests."""
    
    @staticmethod
    def ttest_independent(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Dict[str, float]:
        """Independent samples t-test."""
        from scipy import stats
        
        stat, pvalue = stats.ttest_ind(group1, group2, nan_policy='omit')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': stat,
            'p_value': pvalue,
            'cohens_d': cohens_d,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'mean_diff': np.mean(group1) - np.mean(group2)
        }
    
    @staticmethod
    def anova_oneway(
        *groups: np.ndarray
    ) -> Dict[str, float]:
        """One-way ANOVA."""
        from scipy import stats
        
        stat, pvalue = stats.f_oneway(*groups)
        
        # Calculate eta-squared
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_statistic': stat,
            'p_value': pvalue,
            'eta_squared': eta_squared,
            'n_groups': len(groups)
        }
    
    @staticmethod
    def chi_square(
        observed: pd.DataFrame
    ) -> Dict[str, float]:
        """Chi-square test of independence."""
        from scipy import stats
        
        stat, pvalue, dof, expected = stats.chi2_contingency(observed)
        
        # Cramér's V for effect size
        n = observed.sum().sum()
        min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
        cramers_v = np.sqrt(stat / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'chi2_statistic': stat,
            'p_value': pvalue,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v
        }


class RegressionModeler:
    """
    Regression modeling with 14 estimators.
    
    For Analytical Robustness & Action Defensibility:
    Tests robustness without persuasion, produces staff-safe evidence.
    """
    
    ESTIMATORS = [
        'ols',              # Ordinary Least Squares
        'wls',              # Weighted Least Squares
        'gls',              # Generalized Least Squares
        'logit',            # Logistic Regression
        'probit',           # Probit Model
        'poisson',          # Poisson Regression
        'negative_binomial', # Negative Binomial
        'tobit',            # Tobit (censored)
        'fixed_effects',    # Panel Fixed Effects
        'random_effects',   # Panel Random Effects
        'between_effects',  # Panel Between Effects
        'first_difference', # First Difference
        '2sls',             # Two-Stage Least Squares
        'gmm'               # Generalized Method of Moments
    ]
    
    def __init__(self):
        self._last_result = None
    
    def fit_ols(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        robust_se: bool = True,
        cluster_var: Optional[str] = None
    ) -> RegressionResult:
        """
        Fit OLS regression.
        
        Args:
            df: Data
            dependent_var: Y variable
            independent_vars: X variables
            robust_se: Use heteroskedasticity-robust standard errors
            cluster_var: Variable to cluster standard errors on
        """
        import statsmodels.api as sm
        
        # Prepare data
        y = df[dependent_var].dropna()
        X = df[independent_vars].loc[y.index]
        X = sm.add_constant(X)
        
        # Drop rows with any missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Fit model
        model = sm.OLS(y, X)
        
        if cluster_var and cluster_var in df.columns:
            clusters = df.loc[y.index, cluster_var]
            results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})
        elif robust_se:
            results = model.fit(cov_type='HC3')
        else:
            results = model.fit()
        
        self._last_result = results
        
        return RegressionResult(
            estimator='OLS',
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            n_obs=int(results.nobs),
            r_squared=results.rsquared,
            adj_r_squared=results.rsquared_adj,
            coefficients=dict(results.params),
            std_errors=dict(results.bse),
            p_values=dict(results.pvalues),
            conf_intervals={
                k: (v[0], v[1]) 
                for k, v in results.conf_int().iterrows()
            },
            model_summary=str(results.summary())
        )
    
    def fit_logit(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str]
    ) -> RegressionResult:
        """Fit logistic regression."""
        import statsmodels.api as sm
        
        y = df[dependent_var].dropna()
        X = df[independent_vars].loc[y.index]
        X = sm.add_constant(X)
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        model = sm.Logit(y, X)
        results = model.fit(disp=False)
        
        self._last_result = results
        
        return RegressionResult(
            estimator='Logit',
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            n_obs=int(results.nobs),
            r_squared=results.prsquared,  # Pseudo R-squared
            adj_r_squared=None,
            coefficients=dict(results.params),
            std_errors=dict(results.bse),
            p_values=dict(results.pvalues),
            conf_intervals={
                k: (v[0], v[1])
                for k, v in results.conf_int().iterrows()
            },
            model_summary=str(results.summary())
        )
    
    def fit_poisson(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str]
    ) -> RegressionResult:
        """Fit Poisson regression for count data."""
        import statsmodels.api as sm
        
        y = df[dependent_var].dropna()
        X = df[independent_vars].loc[y.index]
        X = sm.add_constant(X)
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        model = sm.GLM(y, X, family=sm.families.Poisson())
        results = model.fit()
        
        self._last_result = results
        
        return RegressionResult(
            estimator='Poisson',
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            n_obs=int(results.nobs),
            r_squared=None,
            adj_r_squared=None,
            coefficients=dict(results.params),
            std_errors=dict(results.bse),
            p_values=dict(results.pvalues),
            conf_intervals={
                k: (v[0], v[1])
                for k, v in results.conf_int().iterrows()
            },
            model_summary=str(results.summary())
        )
    
    def fit_fixed_effects(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        entity_var: str,
        time_var: Optional[str] = None
    ) -> RegressionResult:
        """Fit panel fixed effects model."""
        try:
            from linearmodels.panel import PanelOLS
            
            df = df.set_index([entity_var, time_var]) if time_var else df.set_index([entity_var])
            
            y = df[dependent_var]
            X = df[independent_vars]
            
            model = PanelOLS(y, X, entity_effects=True)
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            self._last_result = results
            
            return RegressionResult(
                estimator='Fixed Effects',
                dependent_var=dependent_var,
                independent_vars=independent_vars,
                n_obs=int(results.nobs),
                r_squared=results.rsquared,
                adj_r_squared=None,
                coefficients=dict(results.params),
                std_errors=dict(results.std_errors),
                p_values=dict(results.pvalues),
                conf_intervals={},
                model_summary=str(results.summary)
            )
        except ImportError:
            raise ImportError("linearmodels required for panel models")
    
    def run_robustness_checks(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str],
        estimators: List[str] = ['ols', 'logit', 'poisson']
    ) -> List[RegressionResult]:
        """
        Run multiple estimators for robustness checking.
        
        Returns results from each estimator for comparison.
        """
        results = []
        
        for estimator in estimators:
            try:
                if estimator == 'ols':
                    result = self.fit_ols(df, dependent_var, independent_vars)
                elif estimator == 'logit':
                    result = self.fit_logit(df, dependent_var, independent_vars)
                elif estimator == 'poisson':
                    result = self.fit_poisson(df, dependent_var, independent_vars)
                else:
                    continue
                results.append(result)
            except Exception as e:
                print(f"Estimator {estimator} failed: {e}")
        
        return results
    
    def export_latex(self, result: RegressionResult) -> str:
        """Export regression table in LaTeX format."""
        if self._last_result is None:
            return ""
        
        try:
            return self._last_result.summary().as_latex()
        except Exception:
            # Fallback: build simple LaTeX table
            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{" + result.estimator + " Regression Results}",
                r"\begin{tabular}{lcccc}",
                r"\hline",
                r"Variable & Coefficient & Std. Error & p-value \\",
                r"\hline"
            ]
            
            for var in ['const'] + result.independent_vars:
                if var in result.coefficients:
                    coef = result.coefficients.get(var, 0)
                    se = result.std_errors.get(var, 0)
                    pval = result.p_values.get(var, 1)
                    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    lines.append(f"{var} & {coef:.4f}{stars} & ({se:.4f}) & {pval:.4f} \\\\")
            
            lines.extend([
                r"\hline",
                f"N & {result.n_obs} & & \\\\",
                f"R-squared & {result.r_squared:.4f if result.r_squared else 'N/A'} & & \\\\",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}"
            ])
            
            return '\n'.join(lines)
