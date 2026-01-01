"""
Core NLP Engine
================

Production NLP processing for the Institutional Persistence Dashboard.

Provides:
- Keyword extraction (YAKE, TF-IDF, n-grams)
- Sentiment analysis (VADER, NRC emotions)
- Zero-shot classification for custom labels
- Text preprocessing and normalization
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import pandas as pd
import numpy as np


class KeywordExtractor:
    """
    Extract keywords and n-grams from legislative text.
    
    Detects vocabulary persistence and substitution patterns
    across bills, sessions, and venues.
    """
    
    def __init__(self, language: str = "en"):
        self.language = language
        self._yake_extractor = None
        self._tfidf_vectorizer = None
        self._stop_words = None
        
    def _get_yake(self):
        """Lazy load YAKE extractor."""
        if self._yake_extractor is None:
            import yake
            self._yake_extractor = yake.KeywordExtractor(
                lan=self.language,
                n=3,  # Max n-gram size
                dedupLim=0.7,
                top=50,
                features=None
            )
        return self._yake_extractor
    
    def _get_stop_words(self) -> set:
        """Get stop words for the language."""
        if self._stop_words is None:
            try:
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                self._stop_words = set(ENGLISH_STOP_WORDS)
            except ImportError:
                # Fallback minimal stop words
                self._stop_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                    'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
                    'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'could', 'should', 'may', 'might', 'must',
                    'shall', 'this', 'that', 'these', 'those', 'it', 'its'
                }
        return self._stop_words
    
    def extract_keywords_yake(
        self, 
        text: str, 
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using YAKE (unsupervised).
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples, lower score = more important
        """
        if not text or len(text) < 10:
            return []
        
        extractor = self._get_yake()
        keywords = extractor.extract_keywords(text)
        return keywords[:top_n]
    
    def extract_ngrams(
        self,
        texts: List[str],
        n_range: Tuple[int, int] = (1, 3),
        top_n: int = 50,
        min_df: int = 2
    ) -> pd.DataFrame:
        """
        Extract n-grams with TF-IDF scoring.
        
        Args:
            texts: List of documents
            n_range: (min_n, max_n) for n-grams
            top_n: Number of top n-grams to return
            min_df: Minimum document frequency
            
        Returns:
            DataFrame with ngram, frequency, tfidf_score
        """
        if not texts:
            return pd.DataFrame(columns=['ngram', 'frequency', 'tfidf_score'])
        
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        # TF-IDF for importance scoring
        tfidf = TfidfVectorizer(
            ngram_range=n_range,
            stop_words='english',
            min_df=min_df,
            max_features=1000
        )
        
        try:
            tfidf_matrix = tfidf.fit_transform(texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Get average TF-IDF score across documents
            avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Count frequency
            count = CountVectorizer(
                ngram_range=n_range,
                stop_words='english',
                min_df=min_df,
                vocabulary=feature_names
            )
            count_matrix = count.fit_transform(texts)
            frequencies = np.array(count_matrix.sum(axis=0)).flatten()
            
            # Build result DataFrame
            df = pd.DataFrame({
                'ngram': feature_names,
                'frequency': frequencies,
                'tfidf_score': avg_scores
            })
            
            df = df.sort_values('tfidf_score', ascending=False).head(top_n)
            return df.reset_index(drop=True)
            
        except ValueError:
            return pd.DataFrame(columns=['ngram', 'frequency', 'tfidf_score'])
    
    def detect_vocabulary_persistence(
        self,
        texts_by_period: Dict[str, List[str]],
        min_frequency: int = 3
    ) -> pd.DataFrame:
        """
        Detect vocabulary that persists or changes across time periods.
        
        Args:
            texts_by_period: Dict of {period_label: [texts]}
            min_frequency: Minimum frequency to include
            
        Returns:
            DataFrame showing term presence by period
        """
        period_vocab = {}
        
        for period, texts in texts_by_period.items():
            # Extract terms from all texts in period
            all_text = ' '.join(texts)
            words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
            word_counts = Counter(words)
            
            # Filter by frequency and stop words
            stop_words = self._get_stop_words()
            period_vocab[period] = {
                word: count for word, count in word_counts.items()
                if count >= min_frequency and word not in stop_words
            }
        
        # Find all unique terms
        all_terms = set()
        for vocab in period_vocab.values():
            all_terms.update(vocab.keys())
        
        # Build persistence matrix
        rows = []
        for term in sorted(all_terms):
            row = {'term': term}
            for period in texts_by_period.keys():
                row[period] = period_vocab[period].get(term, 0)
            
            # Calculate persistence score (how many periods it appears in)
            presence = sum(1 for p in texts_by_period.keys() if row[p] > 0)
            row['persistence_score'] = presence / len(texts_by_period)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values('persistence_score', ascending=False)


class SentimentAnalyzer:
    """
    Analyze institutional framing and risk signals in legislative text.
    
    This is NOT public opinion sentiment - it's about procedural safety perception.
    """
    
    def __init__(self):
        self._vader = None
        self._nrc_lexicon = None
        self._zero_shot_classifier = None
    
    def _get_vader(self):
        """Lazy load VADER sentiment analyzer."""
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader
    
    def _get_nrc_lexicon(self) -> Dict[str, List[str]]:
        """Lazy load NRC emotion lexicon."""
        if self._nrc_lexicon is None:
            try:
                from nrclex import NRCLex
                self._nrc_lexicon = NRCLex
            except ImportError:
                self._nrc_lexicon = None
        return self._nrc_lexicon
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Returns:
            Dict with neg, neu, pos, compound scores
        """
        if not text:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        vader = self._get_vader()
        return vader.polarity_scores(text)
    
    def analyze_batch_vader(self, texts: List[str]) -> pd.DataFrame:
        """Analyze multiple texts with VADER."""
        results = []
        for text in texts:
            scores = self.analyze_vader(text)
            results.append(scores)
        return pd.DataFrame(results)
    
    def analyze_nrc_emotions(self, text: str) -> Dict[str, int]:
        """
        Analyze 8 emotion categories using NRC lexicon.
        
        Categories: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
        """
        nrc = self._get_nrc_lexicon()
        if nrc is None:
            return {}
        
        try:
            emotion_obj = nrc(text)
            return dict(emotion_obj.affect_frequencies)
        except Exception:
            return {}
    
    def detect_institutional_framing(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Detect institutional framing patterns.
        
        Returns:
            - framing_type: urgency, neutral, defensive
            - risk_signals: list of detected risk language
            - delay_indicators: language suggesting wait/evidence gathering
        """
        text_lower = text.lower()
        
        # Urgency indicators
        urgency_patterns = [
            r'\bnot later than\b', r'\bimmediately\b', r'\bwithout delay\b',
            r'\bpromptly\b', r'\burgent\b', r'\bcritical\b', r'\bexpedited\b',
            r'\bas soon as practicable\b', r'\bwithin \d+ days?\b'
        ]
        
        # Defensive/cautious indicators
        defensive_patterns = [
            r'\bsubject to\b', r'\bnotwithstanding\b', r'\bexcept as provided\b',
            r'\bto the extent practicable\b', r'\bas appropriate\b',
            r'\bin consultation with\b', r'\bafter consideration of\b',
            r'\bif the secretary determines\b', r'\bmay\b.*\bif\b'
        ]
        
        # Delay/evidence-gathering indicators
        delay_patterns = [
            r'\bstudy\b', r'\breview\b', r'\bassessment\b', r'\breport\b',
            r'\bevaluation\b', r'\bconsultation\b', r'\banalysis\b',
            r'\bexamine\b', r'\binvestigate\b', r'\bdetermine whether\b'
        ]
        
        urgency_matches = sum(1 for p in urgency_patterns if re.search(p, text_lower))
        defensive_matches = sum(1 for p in defensive_patterns if re.search(p, text_lower))
        delay_matches = sum(1 for p in delay_patterns if re.search(p, text_lower))
        
        # Determine framing type
        if urgency_matches > defensive_matches and urgency_matches > delay_matches:
            framing_type = "urgency"
        elif defensive_matches > delay_matches:
            framing_type = "defensive"
        elif delay_matches > 0:
            framing_type = "deliberative"
        else:
            framing_type = "neutral"
        
        # Extract specific risk signals
        risk_signals = []
        for pattern in defensive_patterns:
            matches = re.findall(pattern, text_lower)
            risk_signals.extend(matches)
        
        delay_indicators = []
        for pattern in delay_patterns:
            matches = re.findall(pattern, text_lower)
            delay_indicators.extend(matches)
        
        return {
            'framing_type': framing_type,
            'urgency_score': urgency_matches,
            'defensive_score': defensive_matches,
            'delay_score': delay_matches,
            'risk_signals': list(set(risk_signals)),
            'delay_indicators': list(set(delay_indicators))
        }
    
    def classify_zero_shot(
        self,
        text: str,
        labels: List[str],
        multi_label: bool = True
    ) -> Dict[str, float]:
        """
        Zero-shot classification with custom labels.
        
        Requires transformers library with pipeline.
        """
        if self._zero_shot_classifier is None:
            try:
                from transformers import pipeline
                self._zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            except Exception as e:
                return {label: 0.0 for label in labels}
        
        try:
            result = self._zero_shot_classifier(
                text,
                labels,
                multi_label=multi_label
            )
            return dict(zip(result['labels'], result['scores']))
        except Exception:
            return {label: 0.0 for label in labels}


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    @staticmethod
    def clean_legislative_text(text: str) -> str:
        """Clean legislative text for analysis."""
        if not text:
            return ""
        
        # Remove section numbers at start
        text = re.sub(r'^SEC\.\s*\d+[A-Za-z]*\.?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove USC references
        text = re.sub(r'\d+\s+U\.?S\.?C\.?\s+\d+', '', text)
        
        # Remove amendment markers
        text = re.sub(r'\(\d+\)\s*in\s+subsection\s+\([a-z]\)', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
