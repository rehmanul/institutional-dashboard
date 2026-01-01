"""
Page 6: Graph Export for Lobbying Intelligence
================================================

Convert NLP analysis into nodes and edges for graph databases.

Outputs:
- Nodes: Legislative text units with policy sentiment
- Edges: Relationships (same_concept, sentiment_change, entity_flow)
- Neo4j Cypher: Ready-to-import database statements
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph_builder import GraphBuilder, PolicyNode, PolicyEdge, PolicySentimentClassifier
from core.nlp_engine import KeywordExtractor, SentimentAnalyzer

st.set_page_config(
    page_title="Graph Export",
    page_icon="🔗",
    layout="wide"
)

st.markdown("# 🔗 Graph Export for Lobbying Intelligence")

st.markdown("""
<div style="background: #e2e8f0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>🎯 Purpose:</strong> Convert legislative text into structured nodes and edges 
    for graph databases (Neo4j) to power lobbying intelligence workflows.
</div>
""", unsafe_allow_html=True)

st.markdown("""
### What This Produces

| Output | Description |
|--------|-------------|
| **Nodes** | Legislative text units with policy sentiment vectors |
| **Edges** | Relationships: semantic similarity, sentiment shifts, shared entities |
| **Neo4j Cypher** | Ready-to-import database statements |
""")

st.markdown("---")

# File upload
st.markdown("## 📁 Import Legislative Text")
uploaded_file = st.file_uploader(
    "Upload CSV with 'text' column (and optionally 'bill_id', 'section')",
    type=['csv']
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
        
        # Configuration
        st.markdown("## ⚙️ Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            congress = st.number_input("Congress Session", min_value=100, max_value=130, value=119)
        
        with col2:
            bill_col = st.selectbox(
                "Bill ID Column",
                [None] + list(df.columns),
                index=0 if 'bill_id' not in df.columns else list(df.columns).index('bill_id') + 1
            )
        
        with col3:
            section_col = st.selectbox(
                "Section Column",
                [None] + list(df.columns),
                index=0 if 'section' not in df.columns else list(df.columns).index('section') + 1
            )
        
        # Entity extraction option
        extract_entities = st.checkbox("Extract entities from text", value=True)
        extract_keywords = st.checkbox("Extract keywords from text", value=True)
        
        if st.button("🚀 Build Graph", type="primary"):
            with st.spinner("Building nodes and edges..."):
                builder = GraphBuilder()
                keyword_extractor = KeywordExtractor() if extract_keywords else None
                
                # Add columns if not present
                if bill_col is None:
                    df['bill_id'] = 'BILL'
                    bill_col = 'bill_id'
                
                if section_col is None:
                    df['section'] = df.index.astype(str)
                    section_col = 'section'
                
                # Extract entities and keywords if requested
                if extract_entities or extract_keywords:
                    progress = st.progress(0)
                    for i, (idx, row) in enumerate(df.iterrows()):
                        text = str(row['text'])
                        
                        entities = []
                        keywords = []
                        
                        if extract_keywords and keyword_extractor:
                            kw_results = keyword_extractor.extract_keywords_yake(text, top_n=10)
                            keywords = [kw[0] for kw in kw_results]
                        
                        if extract_entities:
                            # Simple entity extraction
                            import re
                            entity_patterns = [
                                r'Department of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                                r'Secretary of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                                r'[A-Z][a-z]+ Agency',
                                r'[A-Z][a-z]+ Administration',
                                r'[A-Z][a-z]+ Commission',
                                r'[A-Z][a-z]+ Committee'
                            ]
                            for pattern in entity_patterns:
                                entities.extend(re.findall(pattern, text))
                        
                        builder.create_node(
                            text=text,
                            bill_id=str(row[bill_col]),
                            section=str(row[section_col]),
                            congress=congress,
                            entities=list(set(entities)),
                            keywords=keywords
                        )
                        
                        progress.progress((i + 1) / len(df))
                else:
                    builder.from_dataframe(
                        df,
                        text_column='text',
                        bill_column=bill_col,
                        section_column=section_col,
                        congress=congress
                    )
                
                # Detect edges
                st.info("Detecting edges between nodes...")
                builder.detect_all_edges()
                
                st.success(f"✅ Built {len(builder.nodes)} nodes and {len(builder.edges)} edges")
                
                st.markdown("---")
                
                # Display results
                st.markdown("## 📊 Results")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Nodes", "Edges", "Neo4j Cypher", "Analysis"])
                
                with tab1:
                    st.markdown("### Policy Nodes")
                    
                    nodes_df = pd.DataFrame([{
                        'node_id': n.node_id,
                        'bill_id': n.bill_id,
                        'section': n.section,
                        'hostile': f"{n.sentiment['hostile']:.2f}",
                        'procedural': f"{n.sentiment['procedural']:.2f}",
                        'supportive': f"{n.sentiment['supportive']:.2f}",
                        'entities': ', '.join(n.entities[:3]),
                        'keywords': ', '.join(n.keywords[:5])
                    } for n in builder.nodes])
                    
                    st.dataframe(nodes_df, use_container_width=True)
                    
                    # Download nodes JSON
                    st.download_button(
                        "📥 Download Nodes (JSON)",
                        builder.export_nodes_json(),
                        "policy_nodes.json",
                        "application/json"
                    )
                
                with tab2:
                    st.markdown("### Policy Edges")
                    
                    if builder.edges:
                        edges_df = pd.DataFrame([{
                            'source': e.source,
                            'target': e.target,
                            'type': e.edge_type,
                            'weight': f"{e.weight:.2f}",
                            'shared_entities': ', '.join(e.shared_entities[:3])
                        } for e in builder.edges])
                        
                        st.dataframe(edges_df, use_container_width=True)
                        
                        # Download edges JSON
                        st.download_button(
                            "📥 Download Edges (JSON)",
                            builder.export_edges_json(),
                            "policy_edges.json",
                            "application/json"
                        )
                    else:
                        st.info("No edges detected. Upload more related documents to find connections.")
                
                with tab3:
                    st.markdown("### Neo4j Cypher Statements")
                    st.caption("Copy and run in Neo4j Browser to import the graph")
                    
                    cypher = builder.export_neo4j_cypher()
                    st.code(cypher, language="cypher")
                    
                    st.download_button(
                        "📥 Download Cypher Script",
                        cypher,
                        "import_graph.cypher",
                        "text/plain"
                    )
                
                with tab4:
                    st.markdown("### Sentiment Distribution")
                    
                    # Aggregate sentiment
                    sentiment_totals = {
                        'hostile': sum(n.sentiment['hostile'] for n in builder.nodes),
                        'procedural': sum(n.sentiment['procedural'] for n in builder.nodes),
                        'supportive': sum(n.sentiment['supportive'] for n in builder.nodes),
                        'regulatory': sum(n.sentiment['regulatory'] for n in builder.nodes),
                        'neutral': sum(n.sentiment['neutral'] for n in builder.nodes)
                    }
                    
                    import plotly.express as px
                    
                    sent_df = pd.DataFrame([
                        {'Category': k.title(), 'Score': v}
                        for k, v in sentiment_totals.items()
                    ])
                    
                    fig = px.bar(sent_df, x='Category', y='Score', 
                                title='Policy Posture Distribution',
                                color='Category')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Edge type summary
                    if builder.edges:
                        st.markdown("### Edge Type Distribution")
                        edge_types = {}
                        for e in builder.edges:
                            base_type = e.edge_type.split(':')[0]
                            edge_types[base_type] = edge_types.get(base_type, 0) + 1
                        
                        st.dataframe(pd.DataFrame([
                            {'Type': k, 'Count': v}
                            for k, v in edge_types.items()
                        ]))
    
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Upload a CSV file to build a lobbying intelligence graph")
    
    with st.expander("📖 Expected Data Format"):
        st.code("""
# CSV with text and optional bill/section columns:
text,bill_id,section
"The Secretary of Defense shall establish...",S1234,101
"Not later than 180 days, submit report...",S1234,102
"Prohibition on certain foreign contracts...",HR5678,201
        """)
    
    st.markdown("""
    ### Node Schema (JSON-LD compatible)
    ```json
    {
      "node_id": "119_S1234_101",
      "text": "...",
      "bill_id": "S1234",
      "section": "101",
      "congress_session": 119,
      "sentiment": {
        "hostile": 0.75,
        "procedural": 0.10,
        "supportive": 0.05,
        "regulatory": 0.05,
        "neutral": 0.05
      },
      "entities": ["Department of Defense"],
      "keywords": ["supply chain", "procurement"]
    }
    ```
    
    ### Edge Types
    - **same_concept**: Semantic similarity between nodes
    - **sentiment_change**: Policy posture shift (calmed_hostility, increased_support)
    - **entity_flow**: Shared entity reference across nodes
    """)
