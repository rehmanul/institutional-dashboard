"""
Graph Builder
==============

Converts NLP analysis output into nodes and edges for graph databases.

Node Schema:
- Consistent IDs: <session>_<bill>_<section>
- Sentiment vectors: hostile, procedural, supportive
- Entities: agencies, sectors, countries
- Keywords: key phrases

Edge Types:
- same_concept: Semantic similarity
- sentiment_change: Policy posture shift
- entity_flow: Shared entity reference
- amends: Legislative modification
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class PolicyNode:
    """
    A node representing a legislative text unit (section, title, subsection).
    
    Designed for lobbying intelligence workflows.
    """
    node_id: str  # Format: <session>_<bill>_<section>
    text: str
    bill_id: str
    section: str
    congress_session: int
    
    # Sentiment - policy posture (not generic positive/negative)
    sentiment: Dict[str, float] = field(default_factory=lambda: {
        'hostile': 0.0,
        'procedural': 0.0,
        'supportive': 0.0,
        'regulatory': 0.0,
        'neutral': 0.0
    })
    
    # Named entities
    entities: List[str] = field(default_factory=list)
    
    # Key phrases
    keywords: List[str] = field(default_factory=list)
    
    # Topic assignment
    topics: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @staticmethod
    def generate_id(congress: int, bill: str, section: str) -> str:
        """Generate consistent node ID."""
        return f"{congress}_{bill.replace('.', '').replace(' ', '')}_{section}"


@dataclass
class PolicyEdge:
    """
    An edge representing a relationship between two nodes.
    """
    edge_id: str
    source: str  # source node_id
    target: str  # target node_id
    edge_type: str  # same_concept, sentiment_change, entity_flow, amends
    
    # Edge properties
    weight: float = 1.0
    
    # Type-specific metadata
    shared_entities: List[str] = field(default_factory=list)
    shared_keywords: List[str] = field(default_factory=list)
    sentiment_delta: Optional[Dict[str, float]] = None
    similarity_score: Optional[float] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @staticmethod
    def generate_id(source: str, target: str, edge_type: str) -> str:
        """Generate edge ID from source, target, and type."""
        raw = f"{source}_{target}_{edge_type}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


class PolicySentimentClassifier:
    """
    Classifies text into policy-specific sentiment categories.
    
    Categories:
    - Hostile: Adversarial, exclusionary, punitive
    - Procedural: Neutral process language
    - Supportive: Enabling, authorizing, funding
    - Regulatory: Compliance, standards, requirements
    - Neutral: Definitions, findings, technical
    """
    
    def __init__(self):
        self._zero_shot = None
        
        # Policy-specific labels for classification
        self.policy_labels = [
            "hostile and adversarial policy language",
            "neutral procedural language",
            "supportive and enabling policy language",
            "regulatory compliance language",
            "neutral technical definitions"
        ]
        
        self.label_mapping = {
            "hostile and adversarial policy language": "hostile",
            "neutral procedural language": "procedural",
            "supportive and enabling policy language": "supportive",
            "regulatory compliance language": "regulatory",
            "neutral technical definitions": "neutral"
        }
    
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify text into policy sentiment categories.
        
        Returns dict with hostile, procedural, supportive, regulatory, neutral scores.
        """
        # Keyword-based fallback (fast, no models needed)
        scores = self._keyword_classify(text)
        
        # Try zero-shot if available
        try:
            from transformers import pipeline
            if self._zero_shot is None:
                self._zero_shot = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            
            result = self._zero_shot(text[:512], self.policy_labels, multi_label=True)
            
            for label, score in zip(result['labels'], result['scores']):
                mapped = self.label_mapping.get(label, 'neutral')
                scores[mapped] = max(scores[mapped], score)
        except Exception:
            pass  # Use keyword scores
        
        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _keyword_classify(self, text: str) -> Dict[str, float]:
        """Keyword-based classification fallback."""
        text_lower = text.lower()
        
        scores = {
            'hostile': 0.0,
            'procedural': 0.0,
            'supportive': 0.0,
            'regulatory': 0.0,
            'neutral': 0.2  # Default baseline
        }
        
        # Hostile indicators
        hostile_keywords = [
            'prohibit', 'ban', 'restrict', 'exclude', 'deny',
            'penalt', 'sanction', 'terminate', 'revoke', 'foreign adversar'
        ]
        scores['hostile'] = sum(0.1 for k in hostile_keywords if k in text_lower)
        
        # Supportive indicators
        supportive_keywords = [
            'authorize', 'appropriate', 'fund', 'support', 'enable',
            'establish', 'create', 'expand', 'enhance', 'accelerate'
        ]
        scores['supportive'] = sum(0.1 for k in supportive_keywords if k in text_lower)
        
        # Regulatory indicators
        regulatory_keywords = [
            'require', 'comply', 'standard', 'certif', 'report',
            'submit', 'deadline', 'within', 'not later than'
        ]
        scores['regulatory'] = sum(0.1 for k in regulatory_keywords if k in text_lower)
        
        # Procedural indicators
        procedural_keywords = [
            'amend', 'strike', 'insert', 'redesignat', 'section',
            'subsection', 'paragraph', 'subparagraph'
        ]
        scores['procedural'] = sum(0.1 for k in procedural_keywords if k in text_lower)
        
        return scores


class GraphBuilder:
    """
    Build nodes and edges from legislative text analysis.
    """
    
    def __init__(self):
        self.sentiment_classifier = PolicySentimentClassifier()
        self.nodes: List[PolicyNode] = []
        self.edges: List[PolicyEdge] = []
    
    def create_node(
        self,
        text: str,
        bill_id: str,
        section: str,
        congress: int = 119,
        entities: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        topics: Optional[List[str]] = None
    ) -> PolicyNode:
        """Create a policy node from text."""
        node_id = PolicyNode.generate_id(congress, bill_id, section)
        
        sentiment = self.sentiment_classifier.classify(text)
        
        node = PolicyNode(
            node_id=node_id,
            text=text[:1000],  # Truncate for storage
            bill_id=bill_id,
            section=section,
            congress_session=congress,
            sentiment=sentiment,
            entities=entities or [],
            keywords=keywords or [],
            topics=topics or []
        )
        
        self.nodes.append(node)
        return node
    
    def create_edge_semantic_similarity(
        self,
        node1: PolicyNode,
        node2: PolicyNode,
        similarity_score: float,
        shared_keywords: Optional[List[str]] = None
    ) -> Optional[PolicyEdge]:
        """Create edge based on semantic similarity."""
        if similarity_score < 0.3:  # Threshold
            return None
        
        edge = PolicyEdge(
            edge_id=PolicyEdge.generate_id(node1.node_id, node2.node_id, 'same_concept'),
            source=node1.node_id,
            target=node2.node_id,
            edge_type='same_concept',
            weight=similarity_score,
            similarity_score=similarity_score,
            shared_keywords=shared_keywords or []
        )
        
        self.edges.append(edge)
        return edge
    
    def create_edge_sentiment_change(
        self,
        node1: PolicyNode,
        node2: PolicyNode
    ) -> Optional[PolicyEdge]:
        """Create edge based on sentiment transformation."""
        # Calculate sentiment delta
        delta = {}
        for key in node1.sentiment:
            delta[key] = node2.sentiment.get(key, 0) - node1.sentiment.get(key, 0)
        
        # Check if significant change
        max_delta = max(abs(v) for v in delta.values())
        if max_delta < 0.2:  # Threshold
            return None
        
        # Determine change type
        if delta.get('hostile', 0) < -0.2:
            change_type = 'calmed_hostility'
        elif delta.get('supportive', 0) > 0.2:
            change_type = 'increased_support'
        elif delta.get('hostile', 0) > 0.2:
            change_type = 'increased_hostility'
        else:
            change_type = 'posture_shift'
        
        edge = PolicyEdge(
            edge_id=PolicyEdge.generate_id(node1.node_id, node2.node_id, 'sentiment_change'),
            source=node1.node_id,
            target=node2.node_id,
            edge_type=f'sentiment_change:{change_type}',
            weight=max_delta,
            sentiment_delta=delta
        )
        
        self.edges.append(edge)
        return edge
    
    def create_edge_entity_flow(
        self,
        node1: PolicyNode,
        node2: PolicyNode
    ) -> Optional[PolicyEdge]:
        """Create edge based on shared entities."""
        shared = set(node1.entities) & set(node2.entities)
        
        if not shared:
            return None
        
        edge = PolicyEdge(
            edge_id=PolicyEdge.generate_id(node1.node_id, node2.node_id, 'entity_flow'),
            source=node1.node_id,
            target=node2.node_id,
            edge_type='entity_flow',
            weight=len(shared) / max(len(node1.entities), len(node2.entities), 1),
            shared_entities=list(shared)
        )
        
        self.edges.append(edge)
        return edge
    
    def detect_all_edges(self) -> List[PolicyEdge]:
        """Detect all edges between existing nodes."""
        new_edges = []
        
        for i, node1 in enumerate(self.nodes):
            for node2 in self.nodes[i+1:]:
                # Entity flow
                edge = self.create_edge_entity_flow(node1, node2)
                if edge:
                    new_edges.append(edge)
                
                # Sentiment change (only if same bill or related)
                if node1.bill_id == node2.bill_id or \
                   set(node1.topics) & set(node2.topics):
                    edge = self.create_edge_sentiment_change(node1, node2)
                    if edge:
                        new_edges.append(edge)
        
        return new_edges
    
    def export_nodes_json(self) -> str:
        """Export all nodes as JSON array."""
        return json.dumps([n.to_dict() for n in self.nodes], indent=2)
    
    def export_edges_json(self) -> str:
        """Export all edges as JSON array."""
        return json.dumps([e.to_dict() for e in self.edges], indent=2)
    
    def export_neo4j_cypher(self) -> str:
        """Export as Neo4j Cypher statements."""
        lines = []
        
        # Create nodes
        for node in self.nodes:
            props = {
                'text': node.text[:500],
                'bill_id': node.bill_id,
                'section': node.section,
                'congress': node.congress_session,
                'hostile': node.sentiment['hostile'],
                'procedural': node.sentiment['procedural'],
                'supportive': node.sentiment['supportive']
            }
            props_str = ', '.join(f'{k}: {json.dumps(v)}' for k, v in props.items())
            lines.append(f"CREATE (n:PolicySection {{node_id: '{node.node_id}', {props_str}}});")
        
        lines.append("")
        
        # Create edges
        for edge in self.edges:
            props = {'weight': edge.weight}
            if edge.shared_entities:
                props['shared_entities'] = edge.shared_entities
            props_str = ', '.join(f'{k}: {json.dumps(v)}' for k, v in props.items())
            
            edge_label = edge.edge_type.upper().replace(':', '_')
            lines.append(
                f"MATCH (a:PolicySection {{node_id: '{edge.source}'}}), "
                f"(b:PolicySection {{node_id: '{edge.target}'}}) "
                f"CREATE (a)-[:{edge_label} {{{props_str}}}]->(b);"
            )
        
        return '\n'.join(lines)
    
    def from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        bill_column: str = 'bill_id',
        section_column: str = 'section',
        congress: int = 119
    ) -> List[PolicyNode]:
        """Build nodes from a DataFrame."""
        nodes = []
        
        for _, row in df.iterrows():
            text = str(row.get(text_column, ''))
            bill = str(row.get(bill_column, 'UNKNOWN'))
            section = str(row.get(section_column, row.name))
            
            # Extract entities if present
            entities = []
            if 'entities' in row and row['entities']:
                entities = row['entities'] if isinstance(row['entities'], list) else []
            
            # Extract keywords if present
            keywords = []
            if 'keywords' in row and row['keywords']:
                keywords = row['keywords'] if isinstance(row['keywords'], list) else []
            
            node = self.create_node(
                text=text,
                bill_id=bill,
                section=section,
                congress=congress,
                entities=entities,
                keywords=keywords
            )
            nodes.append(node)
        
        return nodes
