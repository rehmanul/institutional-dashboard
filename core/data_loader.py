"""
Data Loader
============

Data loading and integration with System 1 (Bill Extractor).
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class DataLoader:
    """Load and prepare data for the dashboard."""
    
    @staticmethod
    def load_csv(
        file_path: str,
        text_column: str = 'text',
        required_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load CSV file with validation.
        
        Args:
            file_path: Path to CSV
            text_column: Name of text column for NLP
            required_columns: List of required column names
        """
        df = pd.read_csv(file_path)
        
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    @staticmethod
    def load_from_streamlit_upload(uploaded_file) -> pd.DataFrame:
        """Load from Streamlit file uploader."""
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.DataFrame(json.load(uploaded_file))
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    
    @staticmethod
    def load_bill_extractor_output(file_path: str) -> pd.DataFrame:
        """
        Load output from Bill Extractor (System 1).
        
        Converts directives format to dashboard format.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            directives = json.load(f)
        
        df = pd.DataFrame(directives)
        
        # Create text column for NLP from available fields
        text_parts = []
        if 'raw_text' in df.columns:
            text_parts.append(df['raw_text'].fillna(''))
        if 'action' in df.columns:
            text_parts.append(df['action'].fillna(''))
        if 'section' in df.columns:
            text_parts.append(df['section'].fillna(''))
        
        if text_parts:
            df['text'] = ' '.join(str(p) for p in text_parts)
        else:
            df['text'] = ''
        
        return df
    
    @staticmethod
    def load_bd_analysis_output(file_path: str) -> pd.DataFrame:
        """Load BD Section Parser output."""
        df = pd.read_csv(file_path)
        
        # Create text column from section content
        text_parts = []
        if 'section_title' in df.columns:
            text_parts.append(df['section_title'].fillna(''))
        if 'raw_text_reference' in df.columns:
            text_parts.append(df['raw_text_reference'].fillna(''))
        
        if text_parts:
            df['text'] = text_parts[0]
            for p in text_parts[1:]:
                df['text'] = df['text'] + ' ' + p
        
        return df
    
    @staticmethod
    def prepare_for_nlp(
        df: pd.DataFrame,
        text_column: str = 'text',
        min_length: int = 10
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for NLP analysis.
        
        - Ensures text column exists
        - Removes empty/short texts
        - Cleans text
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found")
        
        # Filter by minimum length
        df = df[df[text_column].str.len() >= min_length].copy()
        
        # Clean text
        df[text_column] = df[text_column].str.strip()
        
        return df
    
    @staticmethod
    def prepare_for_stats(
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for statistical analysis.
        
        - Converts columns to numeric where possible
        - Handles missing values
        """
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


class ExportManager:
    """Export results in various formats."""
    
    @staticmethod
    def to_csv(df: pd.DataFrame, filename: str) -> bytes:
        """Export DataFrame to CSV bytes."""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def to_json(data: Any, filename: str) -> bytes:
        """Export data to JSON bytes."""
        return json.dumps(data, indent=2, default=str).encode('utf-8')
    
    @staticmethod
    def to_latex(content: str, filename: str) -> bytes:
        """Export LaTeX content."""
        return content.encode('utf-8')
