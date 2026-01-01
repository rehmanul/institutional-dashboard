"""
Page 3: Issue Surfaces & Jurisdictional Emergence
===================================================

Problem: Coalition relevance is implicit and invisible.
Solution: Reveals emergent policy domains without coordination.

Topics ≠ advocacy clusters. Topics = where attention is already forming.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.topic_engine import TopicModeler, CONGRESSIONAL_COMMITTEES, POLICY_AREAS
from utils.visualizations import create_bar_chart, create_topic_treemap, create_heatmap

st.set_page_config(
    page_title="Issue Surfaces",
    page_icon="🎯",
    layout="wide"
)

st.markdown("# 🎯 Issue Surfaces & Jurisdictional Emergence")

st.markdown("""
<div style="background: #fed7d7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>❌ Problem:</strong> Coalition relevance is implicit and invisible.
    A capability touches multiple issue areas but no single coalition coordinates.
</div>
<div style="background: #c6f6d5; padding: 1rem; border-radius: 8px;">
    <strong>✅ Solution:</strong> Reveals emergent policy domains without coordination.
    Maps language to multiple issue surfaces.
</div>
""", unsafe_allow_html=True)

st.info("💡 **Note**: Topics ≠ advocacy clusters. Topics = where attention is already forming.")

st.markdown("---")

# File upload
st.markdown("## 📁 Import Data")
uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column")
            st.stop()
        
        st.success(f"✅ Loaded {len(df)} documents")
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("## ⚙️ Analysis Type")
        
        analysis_type = st.selectbox(
            "Select Analysis",
            [
                "BERTopic Discovery (Unsupervised)",
                "Zero-Shot Topic Classification (Custom Labels)",
                "Jurisdictional Overlap Detection"
            ]
        )
        
        if analysis_type == "BERTopic Discovery (Unsupervised)":
            col1, col2 = st.columns(2)
            with col1:
                min_topic_size = st.slider("Minimum topic size", 3, 20, 5)
            with col2:
                nr_topics = st.selectbox("Number of topics", [None, 5, 10, 15, 20, 30])
            
            if st.button("🚀 Discover Topics", type="primary"):
                modeler = TopicModeler()
                texts = df['text'].dropna().tolist()
                
                if len(texts) < min_topic_size * 2:
                    st.error(f"Need at least {min_topic_size * 2} documents for topic modeling")
                    st.stop()
                
                with st.spinner("Running BERTopic analysis (this may take a moment)..."):
                    try:
                        topics, probs = modeler.fit_bertopic(texts, min_topic_size, nr_topics)
                        topic_info = modeler.get_topic_info()
                        
                        st.markdown("## 📊 Results: Discovered Topics")
                        
                        # Topic overview
                        n_topics = len(topic_info[topic_info['Topic'] != -1])
                        n_outliers = (topic_info[topic_info['Topic'] == -1]['Count'].values[0] 
                                     if -1 in topic_info['Topic'].values else 0)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Topics Found", n_topics)
                        col2.metric("Documents", len(texts))
                        col3.metric("Outliers", n_outliers)
                        
                        # Topic treemap
                        fig = create_topic_treemap(topic_info)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Topic details
                        st.markdown("### Topic Details")
                        for topic_id in topic_info['Topic'].unique():
                            if topic_id == -1:
                                continue
                            words = modeler.get_topic_words(topic_id, top_n=10)
                            if words:
                                with st.expander(f"Topic {topic_id}: {', '.join([w[0] for w in words[:3]])}"):
                                    word_df = pd.DataFrame(words, columns=['Word', 'Score'])
                                    st.dataframe(word_df)
                        
                        # Add topics to dataframe
                        result_df = df.copy()
                        result_df['topic'] = topics[:len(result_df)] if topics else None
                        
                        csv = result_df.to_csv(index=False)
                        st.download_button("📥 Download with Topics", csv, "topics.csv", "text/csv")
                        
                    except ImportError as e:
                        st.error(f"BERTopic not installed: {e}")
                        st.code("pip install bertopic sentence-transformers")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif analysis_type == "Zero-Shot Topic Classification (Custom Labels)":
            st.markdown("### Define Topic Labels")
            
            use_preset = st.checkbox("Use preset policy areas", value=True)
            
            if use_preset:
                labels = st.multiselect(
                    "Select policy areas",
                    POLICY_AREAS,
                    default=POLICY_AREAS[:5]
                )
            else:
                labels_text = st.text_area(
                    "Enter custom labels (one per line)",
                    "Defense Procurement\nCybersecurity\nArtificial Intelligence\nSupply Chain"
                )
                labels = [l.strip() for l in labels_text.split('\n') if l.strip()]
            
            if labels and st.button("🚀 Classify Topics", type="primary"):
                modeler = TopicModeler()
                texts = df['text'].dropna().tolist()
                
                with st.spinner("Running zero-shot classification (requires internet for first run)..."):
                    try:
                        result_df = modeler.classify_zero_shot(texts, labels)
                        
                        st.markdown("## 📊 Results: Topic Classification")
                        
                        # Aggregate scores
                        avg_scores = result_df.mean().sort_values(ascending=False)
                        
                        score_df = pd.DataFrame({
                            'Topic': avg_scores.index,
                            'Average Score': avg_scores.values
                        })
                        
                        fig = create_bar_chart(score_df, x='Topic', y='Average Score', 
                                              title='Topic Prevalence')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Heatmap
                        fig = create_heatmap(result_df.T, title='Document-Topic Matrix')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download
                        full_result = pd.concat([df.reset_index(drop=True), result_df], axis=1)
                        csv = full_result.to_csv(index=False)
                        st.download_button("📥 Download Classification", csv, "topic_classification.csv", "text/csv")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif analysis_type == "Jurisdictional Overlap Detection":
            st.markdown("### Committee Jurisdiction Analysis")
            
            committees = st.multiselect(
                "Select committees to analyze",
                CONGRESSIONAL_COMMITTEES,
                default=CONGRESSIONAL_COMMITTEES[:6]
            )
            
            if committees and st.button("🚀 Analyze Jurisdictions", type="primary"):
                modeler = TopicModeler()
                texts = df['text'].dropna().tolist()
                
                with st.spinner("Analyzing committee jurisdictions..."):
                    try:
                        result_df = modeler.detect_jurisdictional_overlap(texts, committees)
                        
                        st.markdown("## 📊 Results: Jurisdictional Overlap")
                        
                        # Heatmap of overlap
                        corr = result_df.corr()
                        fig = create_heatmap(corr, title='Committee Jurisdiction Correlation')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Average relevance per committee
                        avg_relevance = result_df.mean().sort_values(ascending=False)
                        rel_df = pd.DataFrame({
                            'Committee': avg_relevance.index,
                            'Average Relevance': avg_relevance.values
                        })
                        
                        fig = create_bar_chart(rel_df, x='Committee', y='Average Relevance',
                                              title='Committee Relevance to Documents')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download
                        full_result = pd.concat([df.reset_index(drop=True), result_df], axis=1)
                        csv = full_result.to_csv(index=False)
                        st.download_button("📥 Download Analysis", csv, "jurisdiction_overlap.csv", "text/csv")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👆 Upload a CSV file to begin analysis")
