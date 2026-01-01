"""
Institutional Persistence Dashboard
====================================

A Streamlit application for analyzing policy persistence across institutional change.

Core Problem:
    "Policy relevance does not persist across institutional change, forcing innovators 
    to repeatedly re-translate the same capabilities across congressional sessions, 
    venues, and vehicles under severe time constraints."

Modules:
    1. Vocabulary Persistence & Legislative Legibility (Text2Keywords)
    2. Institutional Framing & Risk Signals (Text2Sentiment)
    3. Issue Surfaces & Jurisdictional Emergence (Text2Topics)
    4. Baseline Conditions & Descriptive Evidence (StatsDashboard)
    5. Analytical Robustness & Action Defensibility (StatsModeling)

Usage:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Institutional Persistence Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Problem statement box */
    .problem-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .problem-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .problem-text {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Module cards */
    .module-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .module-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .module-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .module-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.3rem;
    }
    
    .module-problem {
        font-size: 0.85rem;
        color: #e53e3e;
        margin-bottom: 0.5rem;
    }
    
    .module-solution {
        font-size: 0.9rem;
        color: #38a169;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Main content
def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ Institutional Continuity & Replay</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Procedural Memory for Durable Policy Relevance</p>', unsafe_allow_html=True)
    
    # Core problem statement
    st.markdown("""
    <div class="problem-box">
        <div class="problem-title">📋 Core Problem</div>
        <div class="problem-text">
            Policy relevance does not persist across institutional change, forcing innovators 
            to repeatedly re-translate the same capabilities across congressional sessions, 
            venues, and vehicles under severe time constraints.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Module overview
    st.markdown("## 🔧 Analysis Modules")
    st.markdown("Each module addresses a specific point where institutional persistence breaks down.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">📝</div>
            <div class="module-title">Vocabulary Persistence & Legislative Legibility</div>
            <div class="module-problem">❌ Language does not persist across bills, sessions, or venues</div>
            <div class="module-solution">✅ Detects vocabulary continuity and substitution patterns</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">🎯</div>
            <div class="module-title">Issue Surfaces & Jurisdictional Emergence</div>
            <div class="module-problem">❌ Coalition relevance is implicit and invisible</div>
            <div class="module-solution">✅ Reveals emergent policy domains and jurisdiction overlap</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">🔬</div>
            <div class="module-title">Analytical Robustness & Action Defensibility</div>
            <div class="module-problem">❌ Action is analytically unsafe under scrutiny</div>
            <div class="module-solution">✅ Tests robustness with 14 statistical estimators</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">⚠️</div>
            <div class="module-title">Institutional Framing & Risk Signals</div>
            <div class="module-problem">❌ Institutional tone and risk framing are opaque</div>
            <div class="module-solution">✅ Detects procedural safety perception and delay signals</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">📊</div>
            <div class="module-title">Baseline Conditions & Descriptive Evidence</div>
            <div class="module-problem">❌ Evidence is anecdotal, non-reusable, or non-defensive</div>
            <div class="module-solution">✅ Establishes reusable baseline patterns</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting started
    st.markdown("## 🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Step 1: Prepare Data**
        
        Upload a CSV file with a `text` column containing legislative text, 
        or use output from the Bill Extractor system.
        """)
    
    with col2:
        st.info("""
        **Step 2: Select Module**
        
        Navigate to a module using the sidebar. Each module addresses 
        a specific institutional persistence failure.
        """)
    
    with col3:
        st.info("""
        **Step 3: Analyze & Export**
        
        Run analysis on your data, visualize results, and export 
        findings in CSV, PNG, or LaTeX format.
        """)
    
    # System integration note
    st.markdown("---")
    st.markdown("## 🔗 System Integration")
    
    with st.expander("Integration with Bill Extractor (System 1)", expanded=False):
        st.code("""
# Export directives from Bill Extractor to dashboard format
from extractor.bill_extractor import BillExtractor
import pandas as pd

# Extract from legislation
extractor = BillExtractor()
directives = extractor.extract_from_file("ndaa_bill.html")

# Convert to dashboard input format
df = pd.DataFrame([d.to_dict() for d in directives])

# Prepare text column for NLP modules
df['text'] = df['raw_text'].fillna('') + ' ' + df['action'].fillna('')

# Save for dashboard import
df.to_csv("dashboard_input.csv", index=False)
        """, language="python")
    
    # Footer
    st.markdown("""
    <div class="footer">
        Institutional Persistence Dashboard | Built for Durable Policy Relevance<br>
        Powered by VADER, BERTopic, statsmodels, and linearmodels
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
