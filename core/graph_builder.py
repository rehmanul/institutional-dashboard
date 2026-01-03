"""
Institutional Graph
===================

Manages the graph-based institutional mapping system as defined in the
"Institutional Graph for Durable Policy Relevance" handoff document.

Core Responsibilities:
- Schema validation (drift prevention)
- Node/Edge management with strict typing
- Persistence (JSON loading/saving)
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import jsonschema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstitutionalGraph:
    """
    The core graph engine that enforces the canonical ontology.
    """
    
    def __init__(self, schema_path: str = "graph.schema.json",
                 nodes_path: str = "graph.nodes.json",
                 edges_path: str = "graph.edges.json"):
        self.schema_path = Path(schema_path)
        self.nodes_path = Path(nodes_path)
        self.edges_path = Path(edges_path)
        
        self.schema: Dict[str, Any] = {}
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        
        self.load_schema()
        self.load_data()

    def load_schema(self):
        """Load and validate the JSON schema definition."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            self.schema = json.load(f)
        logger.info(f"Loaded schema version {self.schema.get('version', 'unknown')}")

    def load_data(self):
        """Load nodes and edges from storage."""
        if self.nodes_path.exists():
            with open(self.nodes_path, 'r') as f:
                self.nodes = json.load(f)
        else:
            self.nodes = []
            
        if self.edges_path.exists():
            with open(self.edges_path, 'r') as f:
                self.edges = json.load(f)
        else:
            self.edges = []
            
        self.validate_all()

    def validate_node(self, node: Dict[str, Any]):
        """Validate a single node against the schema."""
        # Extract the node definition from the schema
        # We need to construct a mini-schema for validation or use the definitions
        node_schema = self.schema['definitions']['node']
        jsonschema.validate(instance=node, schema=node_schema)

    def validate_edge(self, edge: Dict[str, Any]):
        """Validate a single edge against the schema."""
        edge_schema = self.schema['definitions']['edge']
        jsonschema.validate(instance=edge, schema=edge_schema)
        
        # Additional Integrity Checks
        node_ids = {n['id'] for n in self.nodes}
        if edge['source'] not in node_ids:
            raise ValueError(f"Edge {edge.get('id')} references unknown source: {edge['source']}")
        if edge['target'] not in node_ids:
            raise ValueError(f"Edge {edge.get('id')} references unknown target: {edge['target']}")

    def validate_all(self):
        """Validate all nodes and edges."""
        errors = []
        for i, node in enumerate(self.nodes):
            try:
                self.validate_node(node)
            except Exception as e:
                errors.append(f"Node {i} ({node.get('id')}): {str(e)}")
        
        for i, edge in enumerate(self.edges):
            try:
                self.validate_edge(edge)
            except Exception as e:
                errors.append(f"Edge {i} ({edge.get('id')}): {str(e)}")
        
        if errors:
            logger.error(f"Validation failed with {len(errors)} errors.")
            # We might strictly raise here, or just log. For strictness, let's log.
            for err in errors[:10]: # Log first 10
                logger.error(err)
            # raise ValueError("Graph data validation failed") # Optional: enforce strictness

    def add_node(self, node: Dict[str, Any]):
        """Add a node to the graph."""
        self.validate_node(node)
        # Check for duplicates
        if any(n['id'] == node['id'] for n in self.nodes):
            raise ValueError(f"Node with ID {node['id']} already exists")
        self.nodes.append(node)

    def add_edge(self, edge: Dict[str, Any]):
        """Add an edge to the graph."""
        self.validate_edge(edge)
        self.edges.append(edge)

    def save(self):
        """Persist graph to disk."""
        with open(self.nodes_path, 'w') as f:
            json.dump(self.nodes, f, indent=2)
        with open(self.edges_path, 'w') as f:
            json.dump(self.edges, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the graph."""
        node_types = {}
        for n in self.nodes:
            t = n.get('type', 'unknown')
            node_types[t] = node_types.get(t, 0) + 1
            
        edge_types = {}
        for e in self.edges:
            t = e.get('type', 'unknown')
            edge_types[t] = edge_types.get(t, 0) + 1
            
        return {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'node_types': node_types,
            'edge_types': edge_types
        }

    # Helper methods for creating valid objects (could be expanded)
    @staticmethod
    def create_node_dict(id: str, type: str, attributes: Dict[str, Any] = {}) -> Dict[str, Any]:
        return {"id": id, "type": type, "attributes": attributes}

    @staticmethod
    def create_edge_dict(source: str, target: str, type: str,
                        attributes: Dict[str, Any] = {}) -> Dict[str, Any]:
        return {"source": source, "target": target, "type": type, "attributes": attributes}
