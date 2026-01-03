"""
Page 6: Institutional Graph Viewer
==================================

Explore the loaded Institutional Graph nodes and edges.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader

st.set_page_config(
    page_title="Graph Viewer",
    page_icon="🔗",
    layout="wide"
)

st.markdown("# 🔗 Institutional Graph Viewer")

st.markdown("""
<div style="background: #e2e8f0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <strong>🎯 Purpose:</strong> Explore the entities (Nodes) and relationships (Edges)
    in the Institutional Graph.
</div>
""", unsafe_allow_html=True)

try:
    graph = DataLoader.load_institutional_graph()
    stats = graph.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Nodes", stats['node_count'])
    with col2:
        st.metric("Total Edges", stats['edge_count'])

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Nodes", "Edges", "Schema"])

    with tab1:
        st.subheader("Nodes")

        # Filter by type
        node_types = list(stats['node_types'].keys())
        selected_type = st.selectbox("Filter by Node Type", ["All"] + node_types)

        nodes_data = []
        for n in graph.nodes:
            if selected_type == "All" or n['type'] == selected_type:
                flat_node = {'id': n['id'], 'type': n['type']}
                flat_node.update(n.get('attributes', {}))
                nodes_data.append(flat_node)

        if nodes_data:
            st.dataframe(pd.DataFrame(nodes_data), use_container_width=True)
        else:
            st.info("No nodes found.")

    with tab2:
        st.subheader("Edges")

        edge_types = list(stats['edge_types'].keys())
        selected_edge_type = st.selectbox("Filter by Edge Type", ["All"] + edge_types)

        edges_data = []
        for e in graph.edges:
            if selected_edge_type == "All" or e['type'] == selected_edge_type:
                flat_edge = {
                    'id': e.get('id', ''),
                    'source': e['source'],
                    'target': e['target'],
                    'type': e['type']
                }
                flat_edge.update(e.get('attributes', {}))
                edges_data.append(flat_edge)

        if edges_data:
            st.dataframe(pd.DataFrame(edges_data), use_container_width=True)
        else:
            st.info("No edges found.")

    with tab3:
        st.subheader("Schema Definition")
        st.json(graph.schema)

except Exception as e:
    st.error(f"Error loading graph: {str(e)}")
