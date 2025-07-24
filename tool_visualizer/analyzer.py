"""
Core Tool Boundary Analyzer - Pure Python Library

This module provides the main analysis functionality without any web server dependencies.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import json
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


class ToolBoundaryAnalyzer:
    """Main analyzer class for tool boundary detection."""
    
    def __init__(self, use_advanced_models: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            use_advanced_models: Whether to use advanced ML models (requires more dependencies)
        """
        self.use_advanced_models = use_advanced_models and HAS_TRANSFORMERS and HAS_SPACY
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500,
            stop_words='english'
        )
        
        if self.use_advanced_models:
            self._load_advanced_models()
        else:
            print("🚀 Running in lightweight mode (TF-IDF only)")
    
    def _load_advanced_models(self):
        """Load advanced ML models."""
        try:
            print("📡 Loading advanced models...")
            self.sentence_model = self._get_sentence_model()
            self.nlp = self._get_spacy_model()
            print("✅ Advanced models loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load advanced models: {e}")
            print("🚀 Falling back to lightweight mode")
            self.use_advanced_models = False
    
    @lru_cache(maxsize=1)
    def _get_sentence_model(self):
        """Get cached sentence transformer model."""
        if not HAS_TRANSFORMERS:
            return None
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            model.max_seq_length = 256
            return model
        except Exception:
            return None
    
    @lru_cache(maxsize=1)
    def _get_spacy_model(self):
        """Get cached spaCy model."""
        if not HAS_SPACY:
            return None
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def analyze(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze tools for boundary overlaps and similarities.
        
        Args:
            tools: List of tool definitions with 'name' and 'description' fields
            
        Returns:
            Dictionary containing analysis results
        """
        if not tools:
            return {"error": "No tools provided"}
        
        print(f"🔍 Analyzing {len(tools)} tools...")
        
        # Extract descriptions
        descriptions = [tool.get('description', '') for tool in tools]
        
        if self.use_advanced_models:
            return self._advanced_analysis(tools, descriptions)
        else:
            return self._basic_analysis(tools, descriptions)
    
    def _basic_analysis(self, tools: List[Dict], descriptions: List[str]) -> Dict[str, Any]:
        """Basic TF-IDF analysis."""
        try:
            # TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
            similarity_matrix = cosine_similarity(tfidf_matrix).astype(float)
            
            # Find overlaps
            overlaps = self._find_overlaps(tools, similarity_matrix)
            recommendations = self._generate_basic_recommendations(tools, overlaps)
            
            return {
                'similarity_matrix': similarity_matrix.tolist(),
                'overlaps': overlaps,
                'recommendations': recommendations,
                'metadata': {
                    'tool_count': len(tools),
                    'average_similarity': float(np.mean(similarity_matrix)),
                    'method': 'TF-IDF Analysis',
                    'mode': 'lightweight'
                }
            }
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def _advanced_analysis(self, tools: List[Dict], descriptions: List[str]) -> Dict[str, Any]:
        """Advanced analysis with multiple similarity metrics."""
        try:
            # Multiple similarity calculations
            tfidf_matrix = self._calculate_tfidf_similarity(descriptions)
            semantic_matrix = self._calculate_semantic_similarity(descriptions)
            
            # Weighted combination
            combined_matrix = (0.6 * tfidf_matrix + 0.4 * semantic_matrix)
            
            # Find overlaps with advanced analysis
            overlaps = self._find_overlaps_advanced(tools, combined_matrix)
            recommendations = self._generate_advanced_recommendations(tools, overlaps)
            
            return {
                'similarity_matrix': combined_matrix.tolist(),
                'overlaps': overlaps,
                'recommendations': recommendations,
                'metadata': {
                    'tool_count': len(tools),
                    'average_similarity': float(np.mean(combined_matrix)),
                    'method': 'Advanced Multi-dimensional Analysis',
                    'mode': 'advanced'
                }
            }
        except Exception as e:
            return {"error": f"Advanced analysis failed: {e}"}
    
    def _calculate_tfidf_similarity(self, descriptions: List[str]) -> np.ndarray:
        """Calculate TF-IDF similarity matrix."""
        if not descriptions:
            return np.array([[]])
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
            return cosine_similarity(tfidf_matrix).astype(float)
        except:
            n = len(descriptions)
            return np.eye(n)
    
    def _calculate_semantic_similarity(self, descriptions: List[str]) -> np.ndarray:
        """Calculate semantic similarity using sentence transformers."""
        if not descriptions or not self.sentence_model:
            return self._calculate_tfidf_similarity(descriptions)
        
        try:
            embeddings = self.sentence_model.encode(descriptions)
            return cosine_similarity(embeddings).astype(float)
        except Exception:
            return self._calculate_tfidf_similarity(descriptions)
    
    def _find_overlaps(self, tools: List[Dict], similarity_matrix: np.ndarray) -> List[Dict]:
        """Find overlapping tools."""
        overlaps = []
        n = len(tools)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity > 0.3:  # Threshold
                    # Extract common words
                    desc1_words = set(tools[i]['description'].lower().split())
                    desc2_words = set(tools[j]['description'].lower().split())
                    
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
                    common_words = [w for w in (desc1_words & desc2_words) 
                                  if w not in stop_words and len(w) > 2]
                    
                    overlaps.append({
                        'tool1': tools[i]['name'],
                        'tool2': tools[j]['name'],
                        'similarity': float(similarity),
                        'commonWords': common_words[:5],
                        'overlap_type': 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                    })
        
        return sorted(overlaps, key=lambda x: x['similarity'], reverse=True)
    
    def _find_overlaps_advanced(self, tools: List[Dict], similarity_matrix: np.ndarray) -> List[Dict]:
        """Find overlaps with advanced analysis."""
        # This would include the advanced logic from the original app.py
        return self._find_overlaps(tools, similarity_matrix)
    
    def _generate_basic_recommendations(self, tools: List[Dict], overlaps: List[Dict]) -> List[Dict]:
        """Generate basic recommendations."""
        recommendations = []
        
        # High similarity tools
        high_overlaps = [o for o in overlaps if o['similarity'] > 0.7]
        if high_overlaps:
            recommendations.append({
                'type': 'merge',
                'message': 'Consider merging highly similar tools:',
                'items': [f"{o['tool1']} and {o['tool2']} ({o['similarity']*100:.1f}% similar)" 
                         for o in high_overlaps]
            })
        
        return recommendations
    
    def _generate_advanced_recommendations(self, tools: List[Dict], overlaps: List[Dict]) -> List[Dict]:
        """Generate advanced recommendations."""
        # This would include the advanced recommendation logic
        return self._generate_basic_recommendations(tools, overlaps)


# Convenience functions for easy usage
def analyze_tools(tools: List[Dict[str, Any]], use_advanced: bool = True) -> Dict[str, Any]:
    """
    Convenience function to analyze tools.
    
    Args:
        tools: List of tool definitions
        use_advanced: Whether to use advanced ML models
        
    Returns:
        Analysis results
    """
    analyzer = ToolBoundaryAnalyzer(use_advanced_models=use_advanced)
    return analyzer.analyze(tools)


def analyze_from_file(filepath: str, use_advanced: bool = True) -> Dict[str, Any]:
    """
    Analyze tools from a JSON file.
    
    Args:
        filepath: Path to JSON file containing tools
        use_advanced: Whether to use advanced ML models
        
    Returns:
        Analysis results
    """
    try:
        with open(filepath, 'r') as f:
            tools = json.load(f)
        return analyze_tools(tools, use_advanced)
    except Exception as e:
        return {"error": f"Failed to load file: {e}"}
