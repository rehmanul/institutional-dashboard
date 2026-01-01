"""
Page 2: Institutional Framing & Risk Signals
=============================================

Problem: Institutional tone and risk framing are opaque.
Solution: Detects procedural safety perception and delay signals.

This is NOT public opinion sentiment - it's about procedural safety perception.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nlp_engine import SentimentAnalyzer
from utils.visualizations import create_sentiment_bars, create_emotion_radar, create_bar_chart

st.set_page_config(
    page_title="Institutional Framing",
    page_icon="⚠️",
    layout="wide"
)

st.markdown("# ⚠️ Institutional Framing & Risk Signals")

st.markdown("""
<div style="background: #fed7d7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>❌ Problem:</strong> Institutional tone and risk framing are opaque.
    Staff delay action due to perceived risk. Innovators misread silence as opposition.
</div>
<div style="background: #c6f6d5; padding: 1rem; border-radius: 8px;">
    <strong>✅ Solution:</strong> Detects institutional framing (urgency, neutrality, defensiveness).
    Explains delay without attributing bad faith.
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ **Important**: This is NOT public opinion sentiment. This is about procedural safety perception.")

st.markdown("---")

# File upload
st.markdown("## 📁 Import Data")
uploaded_file = st.file_uploader(
    "Upload CSV with 'text' column",
    type=['csv']
)

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
                "VADER Sentiment (Overall Tone)",
                "NRC Emotion Categories (8 emotions)",
                "Institutional Framing Detection (Urgency/Defensive/Deliberative)"
            ]
        )
        
        if st.button("🚀 Run Analysis", type="primary"):
            analyzer = SentimentAnalyzer()
            texts = df['text'].dropna().tolist()
            
            with st.spinner("Analyzing institutional framing..."):
                
                if analysis_type == "VADER Sentiment (Overall Tone)":
                    # Batch VADER analysis
                    results = []
                    for text in texts:
                        scores = analyzer.analyze_vader(text)
                        results.append(scores)
                    
                    sentiment_df = pd.DataFrame(results)
                    
                    st.markdown("## 📊 Results: Sentiment Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = create_sentiment_bars(sentiment_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Summary stats
                        st.markdown("### Summary")
                        st.metric("Average Compound", f"{sentiment_df['compound'].mean():.3f}")
                        st.metric("Positive Documents", f"{(sentiment_df['compound'] > 0.05).sum()}")
                        st.metric("Negative Documents", f"{(sentiment_df['compound'] < -0.05).sum()}")
                    
                    # Add to original dataframe
                    result_df = df.copy()
                    result_df['sentiment_neg'] = sentiment_df['neg'].values[:len(result_df)]
                    result_df['sentiment_neu'] = sentiment_df['neu'].values[:len(result_df)]
                    result_df['sentiment_pos'] = sentiment_df['pos'].values[:len(result_df)]
                    result_df['sentiment_compound'] = sentiment_df['compound'].values[:len(result_df)]
                    
                    st.markdown("### Detailed Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "sentiment_analysis.csv", "text/csv")
                
                elif analysis_type == "NRC Emotion Categories (8 emotions)":
                    # NRC emotion analysis
                    emotion_totals = {}
                    for text in texts:
                        emotions = analyzer.analyze_nrc_emotions(text)
                        for emotion, count in emotions.items():
                            emotion_totals[emotion] = emotion_totals.get(emotion, 0) + count
                    
                    if emotion_totals:
                        st.markdown("## 📊 Results: Emotion Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = create_emotion_radar(emotion_totals)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            emotion_df = pd.DataFrame([
                                {'Emotion': k, 'Count': v}
                                for k, v in sorted(emotion_totals.items(), key=lambda x: -x[1])
                            ])
                            fig = create_bar_chart(emotion_df, x='Emotion', y='Count', title='Emotion Counts')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(emotion_df, use_container_width=True)
                        
                        csv = emotion_df.to_csv(index=False)
                        st.download_button("📥 Download Emotions", csv, "emotion_analysis.csv", "text/csv")
                    else:
                        st.warning("NRC lexicon not available. Install with: pip install nrclex")
                
                elif "Institutional Framing" in analysis_type:
                    # Institutional framing detection
                    results = []
                    for i, text in enumerate(texts):
                        framing = analyzer.detect_institutional_framing(text)
                        framing['text_index'] = i
                        framing['text_preview'] = text[:100] + "..." if len(text) > 100 else text
                        results.append(framing)
                    
                    framing_df = pd.DataFrame(results)
                    
                    st.markdown("## 📊 Results: Institutional Framing")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    framing_counts = framing_df['framing_type'].value_counts()
                    with col1:
                        st.metric("🔴 Urgency", framing_counts.get('urgency', 0))
                    with col2:
                        st.metric("🟡 Defensive", framing_counts.get('defensive', 0))
                    with col3:
                        st.metric("🔵 Deliberative", framing_counts.get('deliberative', 0))
                    
                    # Framing distribution chart
                    framing_chart_df = pd.DataFrame({
                        'Framing Type': framing_counts.index,
                        'Count': framing_counts.values
                    })
                    
                    fig = create_bar_chart(framing_chart_df, x='Framing Type', y='Count', title='Framing Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Risk Signals Detected")
                    
                    # Show documents with highest defensive scores
                    defensive_docs = framing_df.nlargest(10, 'defensive_score')
                    st.dataframe(defensive_docs[['text_preview', 'framing_type', 'defensive_score', 'risk_signals']], 
                                use_container_width=True)
                    
                    st.markdown("### Delay Indicators")
                    delay_docs = framing_df.nlargest(10, 'delay_score')
                    st.dataframe(delay_docs[['text_preview', 'framing_type', 'delay_score', 'delay_indicators']], 
                                use_container_width=True)
                    
                    csv = framing_df.to_csv(index=False)
                    st.download_button("📥 Download Framing Analysis", csv, "institutional_framing.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Upload a CSV file to begin analysis")
