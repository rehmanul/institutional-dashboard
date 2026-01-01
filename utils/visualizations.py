"""
Visualization Utilities
========================

Plotly and wordcloud visualizations for the dashboard.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    orientation: str = 'v'
) -> go.Figure:
    """Create a bar chart."""
    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title,
        color=color,
        orientation=orientation
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_histogram(
    data: pd.Series,
    title: str = "",
    bins: int = 30,
    color: str = '#667eea'
) -> go.Figure:
    """Create a histogram."""
    fig = px.histogram(
        data,
        nbins=bins,
        title=title
    )
    fig.update_traces(marker_color=color)
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title=data.name if hasattr(data, 'name') else "Value",
        yaxis_title="Count"
    )
    return fig


def create_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    trendline: bool = False
) -> go.Figure:
    """Create a scatter plot."""
    fig = px.scatter(
        data,
        x=x,
        y=y,
        title=title,
        color=color,
        trendline='ols' if trendline else None
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5
    )
    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: str = "",
    colorscale: str = 'RdBu_r'
) -> go.Figure:
    """Create a correlation heatmap."""
    fig = px.imshow(
        data,
        title=title,
        color_continuous_scale=colorscale,
        aspect='auto'
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5
    )
    return fig


def create_sentiment_bars(
    sentiment_data: pd.DataFrame,
    title: str = "Sentiment Distribution"
) -> go.Figure:
    """Create sentiment distribution bar chart."""
    # Aggregate sentiment categories
    if 'compound' in sentiment_data.columns:
        positive = (sentiment_data['compound'] > 0.05).sum()
        negative = (sentiment_data['compound'] < -0.05).sum()
        neutral = len(sentiment_data) - positive - negative
        
        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [positive, neutral, negative]
        })
        
        colors = ['#38a169', '#718096', '#e53e3e']
        
        fig = go.Figure(data=[
            go.Bar(
                x=data['Sentiment'],
                y=data['Count'],
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            template='plotly_white',
            xaxis_title="Sentiment",
            yaxis_title="Count"
        )
        
        return fig
    
    return go.Figure()


def create_emotion_radar(
    emotions: Dict[str, float],
    title: str = "Emotion Profile"
) -> go.Figure:
    """Create radar chart for emotion distribution."""
    categories = list(emotions.keys())
    values = list(emotions.values())
    
    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.1 if values else 1])
        ),
        template='plotly_white'
    )
    
    return fig


def create_topic_treemap(
    topic_info: pd.DataFrame,
    title: str = "Topic Hierarchy"
) -> go.Figure:
    """Create treemap for topic distribution."""
    if topic_info.empty or 'Topic' not in topic_info.columns:
        return go.Figure()
    
    # Filter out outlier topic (-1)
    data = topic_info[topic_info['Topic'] != -1].copy()
    
    if data.empty:
        return go.Figure()
    
    fig = px.treemap(
        data,
        path=['Topic'],
        values='Count' if 'Count' in data.columns else None,
        title=title
    )
    
    fig.update_layout(
        template='plotly_white',
        title_x=0.5
    )
    
    return fig


def create_regression_coefficient_plot(
    coefficients: Dict[str, float],
    std_errors: Dict[str, float],
    title: str = "Coefficient Estimates"
) -> go.Figure:
    """Create coefficient plot with confidence intervals."""
    vars_list = [v for v in coefficients.keys() if v != 'const']
    coefs = [coefficients[v] for v in vars_list]
    errors = [std_errors.get(v, 0) * 1.96 for v in vars_list]  # 95% CI
    
    fig = go.Figure()
    
    # Add error bars
    fig.add_trace(go.Scatter(
        x=coefs,
        y=vars_list,
        mode='markers',
        marker=dict(size=10, color='#667eea'),
        error_x=dict(type='data', array=errors, color='#667eea'),
        name='Coefficient'
    ))
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Coefficient Estimate",
        yaxis_title="Variable",
        template='plotly_white'
    )
    
    return fig


def create_wordcloud_image(
    word_freq: Dict[str, float],
    width: int = 800,
    height: int = 400
) -> Optional[Any]:
    """
    Create wordcloud image.
    
    Returns PIL Image or None if wordcloud not available.
    """
    try:
        from wordcloud import WordCloud
        
        wc = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(word_freq)
        
        return wc.to_image()
    except ImportError:
        return None


def create_timeline_chart(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "Timeline"
) -> go.Figure:
    """Create timeline/line chart."""
    fig = px.line(
        data,
        x=date_column,
        y=value_column,
        title=title,
        markers=True
    )
    
    fig.update_layout(
        template='plotly_white',
        title_x=0.5
    )
    
    return fig
