"""
Page 4: Baseline Conditions & Descriptive Evidence
====================================================

Problem: Evidence is anecdotal, non-reusable, or non-defensive.
Solution: Establishes baseline conditions and reusable patterns.

No causal claims - this is about legibility and reuse.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stats_engine import DescriptiveStats, StatisticalTests
from utils.visualizations import create_histogram, create_scatter, create_heatmap, create_bar_chart

st.set_page_config(
    page_title="Baseline Evidence",
    page_icon="📊",
    layout="wide"
)

st.markdown("# 📊 Baseline Conditions & Descriptive Evidence")

st.markdown("""
<div style="background: #fed7d7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>❌ Problem:</strong> Evidence is anecdotal, non-reusable, or non-defensive.
    Data exists but lacks descriptive legitimacy.
</div>
<div style="background: #c6f6d5; padding: 1rem; border-radius: 8px;">
    <strong>✅ Solution:</strong> Establishes baseline conditions.
    Makes patterns reusable across sessions.
</div>
""", unsafe_allow_html=True)

st.info("💡 **Note**: No causal claims. This is about legibility and reuse.")

st.markdown("---")

# File upload
st.markdown("## 📁 Import Data")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        with st.expander("Preview Data"):
            st.dataframe(df.head(20))
        
        st.markdown("---")
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Descriptive Stats",
            "🔗 Correlations",
            "🧪 Statistical Tests",
            "📊 Visualizations"
        ])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with tab1:
            st.markdown("## Descriptive Statistics")
            
            if numeric_cols:
                selected_cols = st.multiselect(
                    "Select numeric columns",
                    numeric_cols,
                    default=numeric_cols[:5]
                )
                
                if selected_cols:
                    stats_df = DescriptiveStats.summary(df, selected_cols)
                    
                    st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)
                    
                    csv = stats_df.to_csv()
                    st.download_button("📥 Download Stats", csv, "descriptive_stats.csv", "text/csv")
            else:
                st.warning("No numeric columns found")
            
            if categorical_cols:
                st.markdown("### Categorical Variables")
                cat_col = st.selectbox("Select categorical column", categorical_cols)
                
                if cat_col:
                    freq_df = DescriptiveStats.frequency_table(df, cat_col)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(freq_df)
                    with col2:
                        chart_df = freq_df.reset_index()
                        chart_df.columns = ['Category', 'Count', 'Percent']
                        fig = create_bar_chart(chart_df.head(10), x='Category', y='Count', 
                                              title=f'{cat_col} Distribution')
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("## Correlation Analysis")
            
            if len(numeric_cols) >= 2:
                method = st.selectbox("Correlation method", ['pearson', 'spearman', 'kendall'])
                
                corr_matrix = DescriptiveStats.correlation_matrix(df[numeric_cols], method)
                
                fig = create_heatmap(corr_matrix, title=f'{method.title()} Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                st.markdown("### Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_pairs_df = pd.DataFrame(corr_pairs)
                corr_pairs_df['Abs Correlation'] = corr_pairs_df['Correlation'].abs()
                corr_pairs_df = corr_pairs_df.sort_values('Abs Correlation', ascending=False)
                
                st.dataframe(corr_pairs_df.head(10))
            else:
                st.warning("Need at least 2 numeric columns for correlation")
        
        with tab3:
            st.markdown("## Statistical Tests")
            
            test_type = st.selectbox(
                "Select test",
                ["Independent T-Test", "One-Way ANOVA", "Chi-Square Test"]
            )
            
            if test_type == "Independent T-Test":
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        num_var = st.selectbox("Numeric variable", numeric_cols)
                    with col2:
                        group_var = st.selectbox("Grouping variable", categorical_cols)
                    
                    if st.button("Run T-Test"):
                        groups = df[group_var].unique()
                        if len(groups) >= 2:
                            group1 = df[df[group_var] == groups[0]][num_var].dropna()
                            group2 = df[df[group_var] == groups[1]][num_var].dropna()
                            
                            result = StatisticalTests.ttest_independent(group1.values, group2.values)
                            
                            st.markdown("### T-Test Results")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("t-statistic", f"{result['t_statistic']:.4f}")
                            col2.metric("p-value", f"{result['p_value']:.4f}")
                            col3.metric("Cohen's d", f"{result['cohens_d']:.4f}")
                            
                            st.markdown(f"""
                            - **{groups[0]} mean**: {result['group1_mean']:.4f}
                            - **{groups[1]} mean**: {result['group2_mean']:.4f}
                            - **Mean difference**: {result['mean_diff']:.4f}
                            """)
                            
                            if result['p_value'] < 0.05:
                                st.success("✅ Statistically significant difference (p < 0.05)")
                            else:
                                st.info("ℹ️ No significant difference (p ≥ 0.05)")
            
            elif test_type == "One-Way ANOVA":
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        num_var = st.selectbox("Numeric variable", numeric_cols, key='anova_num')
                    with col2:
                        group_var = st.selectbox("Grouping variable", categorical_cols, key='anova_group')
                    
                    if st.button("Run ANOVA"):
                        groups = []
                        for g in df[group_var].unique():
                            group_data = df[df[group_var] == g][num_var].dropna().values
                            if len(group_data) > 0:
                                groups.append(group_data)
                        
                        if len(groups) >= 2:
                            result = StatisticalTests.anova_oneway(*groups)
                            
                            st.markdown("### ANOVA Results")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("F-statistic", f"{result['f_statistic']:.4f}")
                            col2.metric("p-value", f"{result['p_value']:.4f}")
                            col3.metric("Eta-squared", f"{result['eta_squared']:.4f}")
            
            elif test_type == "Chi-Square Test":
                if len(categorical_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("Variable 1", categorical_cols)
                    with col2:
                        var2 = st.selectbox("Variable 2", [c for c in categorical_cols if c != var1])
                    
                    if st.button("Run Chi-Square"):
                        contingency = pd.crosstab(df[var1], df[var2])
                        result = StatisticalTests.chi_square(contingency)
                        
                        st.markdown("### Chi-Square Results")
                        col1, col2 = st.columns(2)
                        col1.metric("Chi² statistic", f"{result['chi2_statistic']:.4f}")
                        col2.metric("p-value", f"{result['p_value']:.4f}")
                        
                        st.markdown(f"**Cramér's V**: {result['cramers_v']:.4f}")
                        
                        st.markdown("### Contingency Table")
                        st.dataframe(contingency)
        
        with tab4:
            st.markdown("## Visualizations")
            
            viz_type = st.selectbox("Visualization type", ["Histogram", "Scatter Plot"])
            
            if viz_type == "Histogram" and numeric_cols:
                var = st.selectbox("Select variable", numeric_cols)
                bins = st.slider("Number of bins", 10, 100, 30)
                
                fig = create_histogram(df[var].dropna(), title=f'Distribution of {var}', bins=bins)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("X variable", numeric_cols)
                with col2:
                    y_var = st.selectbox("Y variable", [c for c in numeric_cols if c != x_var])
                
                trendline = st.checkbox("Add trendline")
                
                fig = create_scatter(df, x_var, y_var, 
                                    title=f'{x_var} vs {y_var}', 
                                    trendline=trendline)
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👆 Upload a CSV file to begin analysis")
