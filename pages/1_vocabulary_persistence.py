"""
Page 1: Vocabulary Persistence & Legislative Legibility
=========================================================

Problem: Language does not persist across bills, sessions, or venues.
Solution: Detects vocabulary persistence and substitution patterns.

This is NOT keyword search - this is language continuity detection.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nlp_engine import KeywordExtractor, TextPreprocessor
from core.data_loader import DataLoader
from utils.visualizations import create_bar_chart, create_wordcloud_image

st.set_page_config(
    page_title="Vocabulary Persistence",
    page_icon="📝",
    layout="wide"
)

st.markdown("# 📝 Vocabulary Persistence & Legislative Legibility")

st.markdown("""
<div style="background: #fed7d7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>❌ Problem:</strong> Language does not persist across bills, sessions, or venues.
    Same ideas appear under different wording, making continuity invisible.
</div>
<div style="background: #c6f6d5; padding: 1rem; border-radius: 8px;">
    <strong>✅ Solution:</strong> Detects vocabulary persistence and substitution patterns.
    Shows when language already exists without naming the capability.
</div>
""", unsafe_allow_html=True)

st.info("💡 **Note**: This is NOT keyword search. This is language continuity detection across time and venues.")

st.markdown("---")

# File upload
st.markdown("## 📁 Import Data")
uploaded_file = st.file_uploader(
    "Upload CSV with 'text' column",
    type=['csv'],
    help="CSV file must have a 'text' column containing legislative text"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column")
            st.stop()
        
        st.success(f"✅ Loaded {len(df)} documents")
        
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("## ⚙️ Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Keyword Extraction (YAKE)", "N-gram Discovery (TF-IDF)", "Vocabulary Persistence"]
            )
        
        with col2:
            top_n = st.slider("Number of terms to extract", 10, 100, 30)
        
        # Period column for persistence analysis
        period_column = None
        if analysis_type == "Vocabulary Persistence":
            period_columns = [col for col in df.columns if col != 'text']
            if period_columns:
                period_column = st.selectbox(
                    "Time period column (for tracking across sessions)",
                    period_columns
                )
            else:
                st.warning("Add a column indicating time period/session for persistence analysis")
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing vocabulary..."):
                extractor = KeywordExtractor()
                texts = df['text'].dropna().tolist()
                
                if analysis_type == "Keyword Extraction (YAKE)":
                    # Combine all texts for overall keyword extraction
                    combined_text = ' '.join(texts)
                    keywords = extractor.extract_keywords_yake(combined_text, top_n=top_n)
                    
                    st.markdown("## 📊 Results: Top Keywords")
                    
                    if keywords:
                        kw_df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
                        kw_df['Importance'] = 1 / (kw_df['Score'] + 0.01)  # Lower score = more important
                        kw_df = kw_df.sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = create_bar_chart(
                                kw_df.head(20),
                                x='Importance',
                                y='Keyword',
                                title='Top Keywords by Importance',
                                orientation='h'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Word cloud
                            word_freq = dict(zip(kw_df['Keyword'], kw_df['Importance']))
                            wc_img = create_wordcloud_image(word_freq)
                            if wc_img:
                                st.image(wc_img, caption="Word Cloud")
                        
                        st.dataframe(kw_df, use_container_width=True)
                        
                        # Download
                        csv = kw_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Keywords CSV",
                            csv,
                            "keywords.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No keywords extracted")
                
                elif analysis_type == "N-gram Discovery (TF-IDF)":
                    ngram_df = extractor.extract_ngrams(texts, n_range=(1, 3), top_n=top_n)
                    
                    st.markdown("## 📊 Results: N-gram Analysis")
                    
                    if not ngram_df.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = create_bar_chart(
                                ngram_df.head(20),
                                x='tfidf_score',
                                y='ngram',
                                title='Top N-grams by TF-IDF Score',
                                orientation='h'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = create_bar_chart(
                                ngram_df.head(20),
                                x='frequency',
                                y='ngram',
                                title='Top N-grams by Frequency',
                                orientation='h'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(ngram_df, use_container_width=True)
                        
                        csv = ngram_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download N-grams CSV",
                            csv,
                            "ngrams.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No n-grams extracted")
                
                elif analysis_type == "Vocabulary Persistence" and period_column:
                    # Group texts by period
                    texts_by_period = {}
                    for period in df[period_column].unique():
                        period_texts = df[df[period_column] == period]['text'].dropna().tolist()
                        if period_texts:
                            texts_by_period[str(period)] = period_texts
                    
                    if len(texts_by_period) >= 2:
                        persistence_df = extractor.detect_vocabulary_persistence(
                            texts_by_period,
                            min_frequency=3
                        )
                        
                        st.markdown("## 📊 Results: Vocabulary Persistence Across Periods")
                        
                        if not persistence_df.empty:
                            # Show persistent terms
                            persistent = persistence_df[persistence_df['persistence_score'] == 1.0]
                            st.metric("Fully Persistent Terms", len(persistent))
                            
                            st.markdown("### Terms Present in All Periods")
                            st.dataframe(persistent.head(30), use_container_width=True)
                            
                            st.markdown("### All Terms by Persistence Score")
                            st.dataframe(persistence_df, use_container_width=True)
                            
                            csv = persistence_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Persistence Analysis",
                                csv,
                                "vocabulary_persistence.csv",
                                "text/csv"
                            )
                    else:
                        st.warning("Need at least 2 periods for persistence analysis")
    
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload a CSV file to begin analysis")
    
    with st.expander("📖 Expected Data Format"):
        st.code("""
# CSV with text column:
text,section,congress
"The Secretary of Defense shall...",101,117
"Not later than 180 days after...",102,117
"The Secretary shall establish...",103,118
        """)
