import pytest
import json
import os
from core.graph_builder import InstitutionalGraph

# Create dummy files for testing
@pytest.fixture
def graph_files(tmp_path):
    schema = {
        "definitions": {
            "node": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["TestNode"]}
                },
                "required": ["id", "type"]
            },
            "edge": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string", "enum": ["TestEdge"]}
                },
                "required": ["source", "target", "type"]
            }
        }
    }

    nodes = [{"id": "n1", "type": "TestNode"}]
    edges = [{"source": "n1", "target": "n1", "type": "TestEdge"}]

    schema_file = tmp_path / "schema.json"
    nodes_file = tmp_path / "nodes.json"
    edges_file = tmp_path / "edges.json"

    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    with open(nodes_file, 'w') as f:
        json.dump(nodes, f)
    with open(edges_file, 'w') as f:
        json.dump(edges, f)

    return str(schema_file), str(nodes_file), str(edges_file)

def test_graph_loading(graph_files):
    schema, nodes, edges = graph_files
    g = InstitutionalGraph(schema, nodes, edges)
    stats = g.get_stats()
    assert stats['node_count'] == 1
    assert stats['edge_count'] == 1

def test_validation_error(graph_files):
    schema, nodes, edges = graph_files
    g = InstitutionalGraph(schema, nodes, edges)

    # Invalid node
    with pytest.raises(Exception):
        g.add_node({"id": "n2", "type": "InvalidType"})

def test_integrity_error(graph_files):
    schema, nodes, edges = graph_files
    g = InstitutionalGraph(schema, nodes, edges)

    # Edge referencing unknown node
    with pytest.raises(ValueError):
        g.add_edge({"source": "n1", "target": "unknown", "type": "TestEdge"})
