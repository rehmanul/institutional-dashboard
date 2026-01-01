"""
Topic Engine
=============

Topic modeling for Issue Surfaces & Jurisdictional Emergence.

Provides:
- BERTopic unsupervised topic discovery
- Zero-shot topic classification with custom labels
- Topic hierarchy and intertopic distance mapping
"""

from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np


class TopicModeler:
    """
    Topic modeling for detecting issue surfaces and jurisdictional patterns.
    
    Reveals emergent policy domains without formal coordination.
    """
    
    def __init__(self):
        self._bertopic_model = None
        self._zero_shot_classifier = None
    
    def fit_bertopic(
        self,
        texts: List[str],
        min_topic_size: int = 5,
        nr_topics: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Fit BERTopic model to texts.
        
        Args:
            texts: List of documents
            min_topic_size: Minimum documents per topic
            nr_topics: Target number of topics (None = auto)
            
        Returns:
            (topic_assignments, probabilities)
        """
        if not texts or len(texts) < min_topic_size:
            return [], []
        
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Use smaller embedding model for efficiency
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self._bertopic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                calculate_probabilities=True,
                verbose=False
            )
            
            topics, probs = self._bertopic_model.fit_transform(texts)
            return topics, probs.tolist() if hasattr(probs, 'tolist') else list(probs)
            
        except ImportError as e:
            raise ImportError(f"BERTopic or sentence-transformers not installed: {e}")
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information from fitted model."""
        if self._bertopic_model is None:
            return pd.DataFrame()
        
        return self._bertopic_model.get_topic_info()
    
    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top words for a topic."""
        if self._bertopic_model is None:
            return []
        
        try:
            topic = self._bertopic_model.get_topic(topic_id)
            return topic[:top_n] if topic else []
        except Exception:
            return []
    
    def get_document_topic_matrix(
        self,
        texts: List[str]
    ) -> pd.DataFrame:
        """Get document-topic probability matrix."""
        if self._bertopic_model is None:
            return pd.DataFrame()
        
        try:
            probs = self._bertopic_model.approximate_distribution(texts)
            topic_info = self._bertopic_model.get_topic_info()
            
            # Create column names from topic labels
            columns = [f"Topic_{t}" for t in topic_info['Topic'].values if t != -1]
            
            df = pd.DataFrame(probs, columns=columns[:probs.shape[1]])
            return df
        except Exception:
            return pd.DataFrame()
    
    def get_hierarchical_topics(self) -> Dict[str, Any]:
        """Get hierarchical topic structure."""
        if self._bertopic_model is None:
            return {}
        
        try:
            hier = self._bertopic_model.hierarchical_topics()
            return hier.to_dict() if hasattr(hier, 'to_dict') else {}
        except Exception:
            return {}
    
    def classify_zero_shot(
        self,
        texts: List[str],
        topic_labels: List[str],
        multi_label: bool = True
    ) -> pd.DataFrame:
        """
        Classify texts into predefined topic labels.
        
        Args:
            texts: Documents to classify
            topic_labels: List of topic/category labels
            multi_label: Allow multiple labels per document
            
        Returns:
            DataFrame with topic probabilities per document
        """
        if self._zero_shot_classifier is None:
            try:
                from transformers import pipeline
                self._zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            except ImportError:
                return pd.DataFrame()
        
        results = []
        for text in texts:
            try:
                result = self._zero_shot_classifier(
                    text[:512],  # Truncate for model limit
                    topic_labels,
                    multi_label=multi_label
                )
                row = dict(zip(result['labels'], result['scores']))
                results.append(row)
            except Exception:
                results.append({label: 0.0 for label in topic_labels})
        
        return pd.DataFrame(results)
    
    def detect_jurisdictional_overlap(
        self,
        texts: List[str],
        committee_labels: List[str]
    ) -> pd.DataFrame:
        """
        Detect which committees/jurisdictions each text touches.
        
        Returns matrix of text × committee relevance scores.
        """
        # Use zero-shot classification with committee names
        return self.classify_zero_shot(texts, committee_labels, multi_label=True)


# Predefined Congressional committee labels
CONGRESSIONAL_COMMITTEES = [
    "Armed Services",
    "Appropriations",
    "Budget",
    "Foreign Affairs",
    "Intelligence",
    "Homeland Security",
    "Veterans Affairs",
    "Energy and Commerce",
    "Science and Technology",
    "Small Business",
    "Financial Services",
    "Judiciary",
    "Oversight and Reform"
]


# Predefined policy area labels
POLICY_AREAS = [
    "Defense and National Security",
    "Foreign Policy and International Affairs",
    "Economic Development and Trade",
    "Healthcare and Public Health",
    "Energy and Environment",
    "Technology and Innovation",
    "Infrastructure and Transportation",
    "Education and Workforce",
    "Civil Rights and Liberties",
    "Immigration and Border Security"
]
