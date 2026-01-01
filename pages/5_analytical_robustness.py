"""
Page 5: Analytical Robustness & Action Defensibility
======================================================

Problem: Action is analytically unsafe under scrutiny.
Solution: Tests robustness with multiple estimators.

This does NOT optimize outcomes - it reduces risk of acting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stats_engine import RegressionModeler
from utils.visualizations import create_regression_coefficient_plot

st.set_page_config(
    page_title="Analytical Robustness",
    page_icon="🔬",
    layout="wide"
)

st.markdown("# 🔬 Analytical Robustness & Action Defensibility")

st.markdown("""
<div style="background: #fed7d7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>❌ Problem:</strong> Action is analytically unsafe under scrutiny.
    Correlations collapse under questioning. Staff lack analytical cover.
</div>
<div style="background: #c6f6d5; padding: 1rem; border-radius: 8px;">
    <strong>✅ Solution:</strong> Tests robustness without persuasion.
    Produces staff-safe, exportable evidence.
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ **Important**: This does NOT optimize outcomes. It reduces risk of acting.")

st.markdown("---")

# Supported estimators
ESTIMATORS = {
    'OLS': 'Ordinary Least Squares - standard linear regression',
    'Logit': 'Logistic Regression - for binary outcomes (0/1)',
    'Poisson': 'Poisson Regression - for count outcomes',
    'Fixed Effects': 'Panel Fixed Effects - controls for entity-specific effects'
}

# File upload
st.markdown("## 📁 Import Data")
uploaded_file = st.file_uploader("Upload CSV with outcome and predictor variables", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        with st.expander("Preview Data"):
            st.dataframe(df.head(20))
        
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for regression")
            st.stop()
        
        st.markdown("## ⚙️ Model Specification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dependent_var = st.selectbox(
                "Dependent Variable (Y)",
                numeric_cols,
                help="The outcome variable you want to explain"
            )
        
        with col2:
            available_ivs = [c for c in numeric_cols if c != dependent_var]
            independent_vars = st.multiselect(
                "Independent Variables (X)",
                available_ivs,
                default=available_ivs[:3] if len(available_ivs) >= 3 else available_ivs,
                help="Predictor variables"
            )
        
        if not independent_vars:
            st.warning("Select at least one independent variable")
            st.stop()
        
        st.markdown("### Estimator Selection")
        
        estimator = st.selectbox(
            "Primary Estimator",
            list(ESTIMATORS.keys()),
            help="Choose the main regression method"
        )
        
        st.caption(ESTIMATORS[estimator])
        
        # Advanced options
        with st.expander("Advanced Options"):
            robust_se = st.checkbox("Use robust standard errors (HC3)", value=True)
            
            cluster_options = [None] + all_cols
            cluster_var = st.selectbox("Cluster standard errors by", cluster_options)
            
            run_robustness = st.checkbox("Run robustness checks (multiple estimators)", value=False)
        
        if st.button("🚀 Run Regression", type="primary"):
            modeler = RegressionModeler()
            
            with st.spinner("Fitting model..."):
                try:
                    if estimator == 'OLS':
                        result = modeler.fit_ols(
                            df, dependent_var, independent_vars,
                            robust_se=robust_se,
                            cluster_var=cluster_var
                        )
                    elif estimator == 'Logit':
                        result = modeler.fit_logit(df, dependent_var, independent_vars)
                    elif estimator == 'Poisson':
                        result = modeler.fit_poisson(df, dependent_var, independent_vars)
                    else:
                        result = modeler.fit_ols(df, dependent_var, independent_vars, robust_se=robust_se)
                    
                    st.markdown("## 📊 Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Estimator", result.estimator)
                    col2.metric("N", result.n_obs)
                    col3.metric("R²", f"{result.r_squared:.4f}" if result.r_squared else "N/A")
                    col4.metric("Adj R²", f"{result.adj_r_squared:.4f}" if result.adj_r_squared else "N/A")
                    
                    st.markdown("---")
                    
                    # Coefficient table
                    st.markdown("### Coefficient Estimates")
                    
                    coef_data = []
                    for var in ['const'] + independent_vars:
                        if var in result.coefficients:
                            coef_data.append({
                                'Variable': var,
                                'Coefficient': result.coefficients[var],
                                'Std. Error': result.std_errors.get(var, np.nan),
                                'p-value': result.p_values.get(var, np.nan),
                                'Significance': '***' if result.p_values.get(var, 1) < 0.01 
                                               else '**' if result.p_values.get(var, 1) < 0.05 
                                               else '*' if result.p_values.get(var, 1) < 0.1 
                                               else ''
                            })
                    
                    coef_df = pd.DataFrame(coef_data)
                    st.dataframe(
                        coef_df.style.format({
                            'Coefficient': '{:.4f}',
                            'Std. Error': '{:.4f}',
                            'p-value': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.1")
                    
                    # Coefficient plot
                    fig = create_regression_coefficient_plot(
                        result.coefficients,
                        result.std_errors,
                        title='Coefficient Estimates with 95% Confidence Intervals'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Full model summary
                    with st.expander("Full Model Summary"):
                        st.code(result.model_summary)
                    
                    # Robustness checks
                    if run_robustness:
                        st.markdown("---")
                        st.markdown("### 🔄 Robustness Checks")
                        
                        robustness_results = modeler.run_robustness_checks(
                            df, dependent_var, independent_vars,
                            estimators=['ols', 'logit', 'poisson']
                        )
                        
                        if robustness_results:
                            comparison_data = []
                            for r in robustness_results:
                                row = {'Estimator': r.estimator, 'N': r.n_obs}
                                for var in independent_vars:
                                    if var in r.coefficients:
                                        row[var] = f"{r.coefficients[var]:.4f}"
                                        if r.p_values.get(var, 1) < 0.05:
                                            row[var] += '*'
                                comparison_data.append(row)
                            
                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                            st.caption("* indicates p < 0.05")
                    
                    # Export options
                    st.markdown("---")
                    st.markdown("### 📥 Export")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = coef_df.to_csv(index=False)
                        st.download_button(
                            "Download Coefficients (CSV)",
                            csv,
                            "regression_coefficients.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        latex = modeler.export_latex(result)
                        st.download_button(
                            "Download LaTeX Table",
                            latex,
                            "regression_table.tex",
                            "text/plain"
                        )
                    
                    with col3:
                        import json
                        json_str = json.dumps(result.to_dict(), indent=2, default=str)
                        st.download_button(
                            "Download Results (JSON)",
                            json_str,
                            "regression_results.json",
                            "application/json"
                        )
                
                except Exception as e:
                    st.error(f"Model fitting error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.info("👆 Upload a CSV file to begin analysis")
    
    with st.expander("📖 Supported Estimators"):
        for name, desc in ESTIMATORS.items():
            st.markdown(f"**{name}**: {desc}")
