# backend/app.py - Enhanced Tool Boundary Analysis
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from functools import lru_cache
import re
import json
from collections import defaultdict
import spacy
from textblob import TextBlob

app = Flask(__name__)
# Fix CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "ngrok-skip-browser-warning"]
    }
})

# Add response headers to bypass ngrok warning
@app.after_request
def after_request(response):
    # Bypass ngrok browser warning
    response.headers['ngrok-skip-browser-warning'] = 'true'
    # Additional security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Cache the model for better performance
@lru_cache(maxsize=1)
def get_sentence_model():
    try:
        print("Loading sentence transformer model (this may take 60-90 seconds on first run)...")
        # Use a smaller, faster model for production
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        model.max_seq_length = 256  # Reduce memory usage
        print("✅ Sentence transformer model loaded successfully")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load sentence transformer model: {e}")
        print("Continuing with reduced functionality...")
        return None

class ToolAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500,
            stop_words='english'
        )
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️ spaCy model not found. Advanced NLP features disabled.")
            print("To enable: python -m spacy download en_core_web_sm")
            self.nlp = None
        
    def analyze_tools(self, tools):
        """Enhanced analysis with sophisticated boundary detection"""
        descriptions = [tool.get('description', '') for tool in tools]
        
        # Multiple similarity calculations
        tfidf_matrix = self._calculate_tfidf_similarity(descriptions)
        semantic_matrix = self._calculate_semantic_similarity(descriptions)
        
        # Advanced analysis techniques
        intent_matrix = self._calculate_intent_similarity(tools)
        parameter_matrix = self._calculate_parameter_similarity(tools)
        context_matrix = self._calculate_context_similarity(tools)
        
        # Weighted combination for final similarity
        combined_matrix = (
            0.25 * tfidf_matrix + 
            0.35 * semantic_matrix + 
            0.20 * intent_matrix + 
            0.15 * parameter_matrix + 
            0.05 * context_matrix
        )
        
        # Enhanced overlap analysis
        overlaps = self._analyze_overlaps_advanced(combined_matrix, tools)
        recommendations = self._generate_advanced_recommendations(tools, overlaps)
        
        return {
            'similarity_matrix': combined_matrix.tolist(),
            'overlaps': overlaps,
            'recommendations': recommendations,
            'metadata': {
                'tool_count': len(tools),
                'average_similarity': float(np.mean(combined_matrix)),
                'method': 'Multi-dimensional Analysis (TF-IDF + Semantic + Intent + Parameters + Context)',
                'analysis_details': {
                    'tfidf_avg': float(np.mean(tfidf_matrix)),
                    'semantic_avg': float(np.mean(semantic_matrix)),
                    'intent_avg': float(np.mean(intent_matrix)),
                    'parameter_avg': float(np.mean(parameter_matrix)),
                    'context_avg': float(np.mean(context_matrix))
                }
            }
        }
    
    def _calculate_tfidf_similarity(self, descriptions):
        """TF-IDF based similarity"""
        if not descriptions:
            return np.array([[]])
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
            return cosine_similarity(tfidf_matrix).astype(float)
        except:
            # Fallback to simple similarity if TF-IDF fails
            n = len(descriptions)
            return np.eye(n)
    
    def _calculate_semantic_similarity(self, descriptions):
        """Semantic similarity using sentence transformers"""
        if not descriptions:
            return np.array([[]])
        try:
            model = get_sentence_model()
            if model is None:
                print("Falling back to TF-IDF similarity (sentence transformer unavailable)")
                return self._calculate_tfidf_similarity(descriptions)
            embeddings = model.encode(descriptions)
            return cosine_similarity(embeddings).astype(float)
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            # Fallback to TF-IDF only
            return self._calculate_tfidf_similarity(descriptions)
    
    def _calculate_intent_similarity(self, tools):
        """Analyze intent similarity using verb-object patterns and action types"""
        n = len(tools)
        matrix = np.zeros((n, n))
        
        # Extract intents from each tool
        intents = []
        for tool in tools:
            intent = self._extract_intent(tool)
            intents.append(intent)
        
        # Calculate intent similarity
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._compare_intents(intents[i], intents[j])
        
        return matrix
    
    def _extract_intent(self, tool):
        """Extract intent patterns from tool description"""
        desc = tool.get('description', '').lower()
        name = tool.get('name', '').lower()
        
        # Action verbs that indicate intent
        action_verbs = {
            'create': ['create', 'make', 'generate', 'build', 'add', 'insert', 'new'],
            'read': ['read', 'get', 'fetch', 'retrieve', 'find', 'search', 'query', 'list', 'show', 'display'],
            'update': ['update', 'modify', 'change', 'edit', 'alter', 'replace', 'set'],
            'delete': ['delete', 'remove', 'clear', 'destroy', 'drop'],
            'analyze': ['analyze', 'process', 'calculate', 'compute', 'evaluate', 'assess'],
            'transform': ['transform', 'convert', 'translate', 'format', 'parse', 'encode', 'decode'],
            'communicate': ['send', 'receive', 'connect', 'call', 'request', 'post', 'notify'],
            'control': ['start', 'stop', 'pause', 'resume', 'execute', 'run', 'trigger']
        }
        
        # Object domains
        domain_keywords = {
            'data': ['data', 'database', 'record', 'table', 'row', 'column', 'field'],
            'file': ['file', 'document', 'folder', 'directory', 'path', 'storage'],
            'web': ['web', 'http', 'url', 'api', 'endpoint', 'request', 'response'],
            'text': ['text', 'string', 'content', 'message', 'word', 'paragraph'],
            'image': ['image', 'photo', 'picture', 'visual', 'graphic', 'pixel'],
            'user': ['user', 'person', 'profile', 'account', 'authentication'],
            'system': ['system', 'server', 'process', 'service', 'application'],
            'math': ['math', 'calculate', 'number', 'formula', 'equation', 'statistics']
        }
        
        intent = {
            'actions': [],
            'domains': [],
            'complexity': 0
        }
        
        # Find action verbs
        text = f"{name} {desc}"
        for action_type, verbs in action_verbs.items():
            if any(verb in text for verb in verbs):
                intent['actions'].append(action_type)
        
        # Find domains
        for domain_type, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                intent['domains'].append(domain_type)
        
        # Estimate complexity (number of operations, conditions, etc.)
        complexity_indicators = ['and', 'or', 'then', 'if', 'multiple', 'several', 'various']
        intent['complexity'] = sum(1 for indicator in complexity_indicators if indicator in text)
        
        return intent
    
    def _compare_intents(self, intent1, intent2):
        """Compare two intent structures"""
        if not intent1['actions'] and not intent2['actions']:
            return 0.5
        
        # Compare actions
        action_overlap = len(set(intent1['actions']) & set(intent2['actions']))
        action_union = len(set(intent1['actions']) | set(intent2['actions']))
        action_similarity = action_overlap / action_union if action_union > 0 else 0
        
        # Compare domains
        domain_overlap = len(set(intent1['domains']) & set(intent2['domains']))
        domain_union = len(set(intent1['domains']) | set(intent2['domains']))
        domain_similarity = domain_overlap / domain_union if domain_union > 0 else 0
        
        # Compare complexity
        complexity_diff = abs(intent1['complexity'] - intent2['complexity'])
        complexity_similarity = 1 / (1 + complexity_diff)
        
        # Weighted combination
        return 0.5 * action_similarity + 0.3 * domain_similarity + 0.2 * complexity_similarity
    
    def _calculate_parameter_similarity(self, tools):
        """Analyze parameter structure similarity"""
        n = len(tools)
        matrix = np.zeros((n, n))
        
        # Extract parameter signatures
        param_signatures = []
        for tool in tools:
            params = tool.get('parameters', {})
            signature = self._extract_parameter_signature(params)
            param_signatures.append(signature)
        
        # Calculate parameter similarity
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._compare_parameter_signatures(param_signatures[i], param_signatures[j])
        
        return matrix
    
    def _extract_parameter_signature(self, params):
        """Extract parameter signature from tool parameters"""
        if not params:
            return {'types': [], 'required': [], 'optional': [], 'structure': 'simple'}
        
        signature = {
            'types': [],
            'required': [],
            'optional': [],
            'structure': 'simple'
        }
        
        # Handle different parameter formats
        if isinstance(params, dict):
            if 'properties' in params:  # JSON schema format
                props = params.get('properties', {})
                required = params.get('required', [])
                
                for prop_name, prop_def in props.items():
                    prop_type = prop_def.get('type', 'unknown')
                    signature['types'].append(prop_type)
                    
                    if prop_name in required:
                        signature['required'].append(prop_name)
                    else:
                        signature['optional'].append(prop_name)
                
                if len(props) > 3:
                    signature['structure'] = 'complex'
            else:
                # Simple parameter dict
                for key, value in params.items():
                    if isinstance(value, dict):
                        signature['structure'] = 'complex'
                    signature['types'].append(type(value).__name__)
        
        return signature
    
    def _compare_parameter_signatures(self, sig1, sig2):
        """Compare parameter signatures"""
        if not sig1['types'] and not sig2['types']:
            return 0.8  # Both have no parameters
        
        # Compare types
        type_overlap = len(set(sig1['types']) & set(sig2['types']))
        type_union = len(set(sig1['types']) | set(sig2['types']))
        type_similarity = type_overlap / type_union if type_union > 0 else 0
        
        # Compare required vs optional pattern
        req_diff = abs(len(sig1['required']) - len(sig2['required']))
        opt_diff = abs(len(sig1['optional']) - len(sig2['optional']))
        param_count_similarity = 1 / (1 + req_diff + opt_diff)
        
        # Compare structure complexity
        structure_similarity = 1.0 if sig1['structure'] == sig2['structure'] else 0.5
        
        return 0.4 * type_similarity + 0.4 * param_count_similarity + 0.2 * structure_similarity
    
    def _calculate_context_similarity(self, tools):
        """Analyze context and usage pattern similarity"""
        n = len(tools)
        matrix = np.zeros((n, n))
        
        # Extract context patterns
        contexts = []
        for tool in tools:
            context = self._extract_context(tool)
            contexts.append(context)
        
        # Calculate context similarity
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._compare_contexts(contexts[i], contexts[j])
        
        return matrix
    
    def _extract_context(self, tool):
        """Extract context patterns from tool description"""
        desc = tool.get('description', '').lower()
        name = tool.get('name', '').lower()
        
        # Context indicators
        context_patterns = {
            'async': ['async', 'asynchronous', 'background', 'queue', 'schedule'],
            'batch': ['batch', 'bulk', 'multiple', 'all', 'mass', 'group'],
            'realtime': ['real', 'time', 'live', 'instant', 'immediate', 'now'],
            'secure': ['secure', 'auth', 'permission', 'token', 'credential', 'private'],
            'external': ['external', 'third', 'party', 'api', 'service', 'remote'],
            'internal': ['internal', 'local', 'cache', 'memory', 'storage', 'database'],
            'user_facing': ['user', 'interface', 'display', 'show', 'present', 'ui'],
            'system_level': ['system', 'admin', 'config', 'setting', 'management']
        }
        
        context = {
            'patterns': [],
            'data_flow': 'unknown',
            'user_interaction': False
        }
        
        text = f"{name} {desc}"
        
        # Find context patterns
        for pattern_type, keywords in context_patterns.items():
            if any(keyword in text for keyword in keywords):
                context['patterns'].append(pattern_type)
        
        # Determine data flow
        if any(word in text for word in ['input', 'receive', 'get', 'fetch']):
            if any(word in text for word in ['output', 'send', 'return', 'provide']):
                context['data_flow'] = 'bidirectional'
            else:
                context['data_flow'] = 'input'
        elif any(word in text for word in ['output', 'send', 'return', 'provide']):
            context['data_flow'] = 'output'
        
        # Check user interaction
        context['user_interaction'] = any(word in text for word in ['user', 'interact', 'click', 'select', 'choose'])
        
        return context
    
    def _compare_contexts(self, ctx1, ctx2):
        """Compare context patterns"""
        # Compare patterns
        pattern_overlap = len(set(ctx1['patterns']) & set(ctx2['patterns']))
        pattern_union = len(set(ctx1['patterns']) | set(ctx2['patterns']))
        pattern_similarity = pattern_overlap / pattern_union if pattern_union > 0 else 0
        
        # Compare data flow
        flow_similarity = 1.0 if ctx1['data_flow'] == ctx2['data_flow'] else 0.3
        
        # Compare user interaction
        interaction_similarity = 1.0 if ctx1['user_interaction'] == ctx2['user_interaction'] else 0.5
        
        return 0.5 * pattern_similarity + 0.3 * flow_similarity + 0.2 * interaction_similarity
    
    def _analyze_overlaps_advanced(self, similarity_matrix, tools):
        """Advanced overlap analysis with detailed confusion detection"""
        overlaps = []
        n = len(tools)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity > 0.25:  # Lower threshold for more detailed analysis
                    
                    # Detailed analysis of the overlap
                    tool1 = tools[i]
                    tool2 = tools[j]
                    
                    # Extract detailed overlap information
                    overlap_analysis = self._detailed_overlap_analysis(tool1, tool2)
                    
                    # Determine confusion risk
                    confusion_risk = self._calculate_confusion_risk(similarity, overlap_analysis)
                    
                    overlaps.append({
                        'tool1': tool1['name'],
                        'tool2': tool2['name'],
                        'similarity': float(similarity),
                        'commonWords': overlap_analysis['common_words'][:5],
                        'overlap_type': 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low',
                        'confusion_risk': confusion_risk,
                        'overlap_details': {
                            'intent_overlap': overlap_analysis['intent_overlap'],
                            'parameter_overlap': overlap_analysis['parameter_overlap'],
                            'context_overlap': overlap_analysis['context_overlap'],
                            'semantic_overlap': overlap_analysis['semantic_overlap']
                        },
                        'disambiguation_suggestions': overlap_analysis['suggestions']
                    })
        
        return sorted(overlaps, key=lambda x: x['confusion_risk'], reverse=True)
    
    def _detailed_overlap_analysis(self, tool1, tool2):
        """Perform detailed analysis of overlap between two tools"""
        desc1 = tool1.get('description', '').lower()
        desc2 = tool2.get('description', '').lower()
        name1 = tool1.get('name', '').lower()
        name2 = tool2.get('name', '').lower()
        
        # Common words analysis
        words1 = set(desc1.split())
        words2 = set(desc2.split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'from', 'with', 'by', 'of', 'as', 'is', 'was', 'are', 'were'}
        common_words = [w for w in (words1 & words2) if w not in stop_words and len(w) > 2]
        
        # Intent overlap
        intent1 = self._extract_intent(tool1)
        intent2 = self._extract_intent(tool2)
        intent_overlap = self._compare_intents(intent1, intent2)
        
        # Parameter overlap
        params1 = tool1.get('parameters', {})
        params2 = tool2.get('parameters', {})
        param_sig1 = self._extract_parameter_signature(params1)
        param_sig2 = self._extract_parameter_signature(params2)
        parameter_overlap = self._compare_parameter_signatures(param_sig1, param_sig2)
        
        # Context overlap
        ctx1 = self._extract_context(tool1)
        ctx2 = self._extract_context(tool2)
        context_overlap = self._compare_contexts(ctx1, ctx2)
        
        # Semantic overlap using sentence transformers
        try:
            model = get_sentence_model()
            emb1 = model.encode([desc1])
            emb2 = model.encode([desc2])
            semantic_overlap = float(cosine_similarity(emb1, emb2)[0][0])
        except:
            semantic_overlap = 0.0
        
        # Generate disambiguation suggestions
        suggestions = self._generate_disambiguation_suggestions(
            tool1, tool2, intent1, intent2, common_words
        )
        
        return {
            'common_words': common_words,
            'intent_overlap': intent_overlap,
            'parameter_overlap': parameter_overlap,
            'context_overlap': context_overlap,
            'semantic_overlap': semantic_overlap,
            'suggestions': suggestions
        }
    
    def _calculate_confusion_risk(self, similarity, overlap_analysis):
        """Calculate the risk of agent confusion between two tools"""
        base_risk = similarity
        
        # Increase risk based on specific overlap types
        if overlap_analysis['intent_overlap'] > 0.7:
            base_risk += 0.2  # High intent overlap is very confusing
        
        if overlap_analysis['parameter_overlap'] > 0.8:
            base_risk += 0.15  # Similar parameters increase confusion
        
        if len(overlap_analysis['common_words']) > 3:
            base_risk += 0.1  # Many common words increase confusion
        
        # Decrease risk if contexts are very different
        if overlap_analysis['context_overlap'] < 0.3:
            base_risk -= 0.1  # Different contexts reduce confusion
        
        return min(1.0, max(0.0, base_risk))
    
    def _generate_disambiguation_suggestions(self, tool1, tool2, intent1, intent2, common_words):
        """Generate suggestions to disambiguate between similar tools"""
        suggestions = []
        
        # Name-based suggestions
        if any(word in tool1['name'].lower() for word in common_words):
            suggestions.append(f"Consider renaming '{tool1['name']}' to be more specific")
        
        if any(word in tool2['name'].lower() for word in common_words):
            suggestions.append(f"Consider renaming '{tool2['name']}' to be more specific")
        
        # Intent-based suggestions
        if intent1['actions'] == intent2['actions']:
            suggestions.append("Tools perform same actions - consider merging or clarifying different use cases")
        
        if intent1['domains'] == intent2['domains']:
            suggestions.append("Tools operate in same domain - specify different data types or contexts")
        
        # Parameter-based suggestions
        suggestions.append("Add distinctive parameters to clarify different use cases")
        
        # Context-based suggestions
        suggestions.append("Specify different contexts or prerequisites in descriptions")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _generate_advanced_recommendations(self, tools, overlaps):
        """Generate advanced recommendations based on sophisticated analysis"""
        recommendations = []
        
        # High confusion risk tools
        high_risk_overlaps = [o for o in overlaps if o['confusion_risk'] > 0.8]
        if high_risk_overlaps:
            recommendations.append({
                'type': 'high_confusion_risk',
                'message': '🚨 High confusion risk - Agent may struggle to distinguish:',
                'items': [f"{o['tool1']} ↔ {o['tool2']} (Risk: {o['confusion_risk']*100:.1f}%)" 
                         for o in high_risk_overlaps]
            })
        
        # Intent disambiguation
        intent_conflicts = [o for o in overlaps if o['overlap_details']['intent_overlap'] > 0.7]
        if intent_conflicts:
            recommendations.append({
                'type': 'intent_clarification',
                'message': '🎯 Intent clarification needed - Similar purposes detected:',
                'items': [f"{o['tool1']} and {o['tool2']} have {o['overlap_details']['intent_overlap']*100:.1f}% intent overlap" 
                         for o in intent_conflicts]
            })
        
        # Parameter structure conflicts
        param_conflicts = [o for o in overlaps if o['overlap_details']['parameter_overlap'] > 0.8]
        if param_conflicts:
            recommendations.append({
                'type': 'parameter_structure',
                'message': '⚙️ Parameter structure too similar - May cause selection confusion:',
                'items': [f"{o['tool1']} and {o['tool2']}" for o in param_conflicts]
            })
        
        # Context clarification
        context_similar = [o for o in overlaps if o['overlap_details']['context_overlap'] > 0.7]
        if context_similar:
            recommendations.append({
                'type': 'context_clarification',
                'message': '📋 Context clarification needed - Similar usage patterns:',
                'items': [f"{o['tool1']} and {o['tool2']}" for o in context_similar]
            })
        
        # Semantic disambiguation
        semantic_conflicts = [o for o in overlaps if o['overlap_details']['semantic_overlap'] > 0.8]
        if semantic_conflicts:
            recommendations.append({
                'type': 'semantic_disambiguation',
                'message': '📝 Semantic disambiguation needed - Descriptions too similar:',
                'items': [f"{o['tool1']} and {o['tool2']}" for o in semantic_conflicts]
            })
        
        # Tool consolidation suggestions
        merge_candidates = [o for o in overlaps if o['similarity'] > 0.9]
        if merge_candidates:
            recommendations.append({
                'type': 'merge_candidates',
                'message': '🔗 Consider merging these nearly identical tools:',
                'items': [f"{o['tool1']} and {o['tool2']} ({o['similarity']*100:.1f}% similar)" 
                         for o in merge_candidates]
            })
        
        return recommendations
        """Find overlapping tools - matches frontend format"""
        overlaps = []
        n = len(tools)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity > 0.3:  # Threshold
                    # Extract common words (simple approach for frontend compatibility)
                    desc1_words = set(tools[i]['description'].lower().split())
                    desc2_words = set(tools[j]['description'].lower().split())
                    
                    # Remove stop words
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'from', 'with', 'by', 'of', 'as', 'is', 'was', 'are', 'were'}
                    common_words = [w for w in (desc1_words & desc2_words) 
                                  if w not in stop_words and len(w) > 2]
                    
                    overlaps.append({
                        'tool1': tools[i]['name'],
                        'tool2': tools[j]['name'],
                        'similarity': float(similarity),
                        'commonWords': common_words[:5],  # Frontend expects this key
                        'overlap_type': 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                    })
        
        return sorted(overlaps, key=lambda x: x['similarity'], reverse=True)
    
    def _generate_recommendations(self, tools, overlaps):
        """Generate recommendations matching frontend format"""
        recommendations = []
        
        # Check for highly similar tools
        high_overlaps = [o for o in overlaps if o['similarity'] > 0.7]
        if high_overlaps:
            recommendations.append({
                'type': 'merge',
                'message': 'Consider merging highly similar tools:',
                'items': [f"{o['tool1']} and {o['tool2']} ({o['similarity']*100:.1f}% similar)" 
                         for o in high_overlaps]
            })
        
        # Check for tools with many overlaps
        overlap_count = {}
        for o in overlaps:
            if o['similarity'] > 0.3:
                overlap_count[o['tool1']] = overlap_count.get(o['tool1'], 0) + 1
                overlap_count[o['tool2']] = overlap_count.get(o['tool2'], 0) + 1
        
        ambiguous_tools = [tool for tool, count in overlap_count.items() if count >= 3]
        if ambiguous_tools:
            recommendations.append({
                'type': 'clarify',
                'message': 'These tools have unclear boundaries with multiple others:',
                'items': ambiguous_tools
            })
        
        # Check for generic naming
        generic_words = ['search', 'get', 'find', 'query', 'fetch', 'retrieve']
        tools_with_generic = [t['name'] for t in tools 
                             if any(word in t['name'].lower() for word in generic_words)]
        
        if len(tools_with_generic) > 1:
            recommendations.append({
                'type': 'rename',
                'message': 'Consider more specific names for these tools:',
                'items': tools_with_generic
            })
        
        return recommendations

# Initialize analyzer
analyzer = ToolAnalyzer()

@app.route('/')
@cross_origin()
def serve_frontend():
    """Serve the frontend HTML file"""
    return send_from_directory('..', 'index.html')

@app.route('/test')
@cross_origin()  
def serve_test():
    """Serve the test page"""
    return send_from_directory('..', 'test.html')

@app.route('/health')
@cross_origin()
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Tool Boundary Analyzer Backend',
        'version': '1.0',
        'endpoints': {
            'analyze': '/api/analyze',
            'health': '/health'
        }
    })

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
@cross_origin()
def analyze():
    """Main analysis endpoint"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        tools = data.get('tools', [])
        
        if not tools:
            return jsonify({'error': 'No tools provided'}), 400
        
        # Validate tool format
        for tool in tools:
            if 'name' not in tool or 'description' not in tool:
                return jsonify({'error': 'Each tool must have name and description'}), 400
        
        print(f"Analyzing {len(tools)} tools...")
        
        # Perform analysis
        result = analyzer.analyze_tools(tools)
        
        print(f"Analysis complete. Found {len(result['overlaps'])} overlaps.")
        
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        error_response = jsonify({'error': f'Analysis failed: {str(e)}'})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

@app.route('/warmup')
@cross_origin()
def warmup():
    """Warmup endpoint to pre-load models"""
    try:
        print("Warming up models...")
        
        # Pre-load sentence transformer
        model = get_sentence_model()
        if model:
            # Test with a simple sentence
            test_embedding = model.encode(["test"])
            print("✅ Sentence transformer warmed up")
        
        # Pre-load spaCy if available
        if analyzer.nlp:
            test_doc = analyzer.nlp("test")
            print("✅ spaCy warmed up")
        
        return jsonify({
            'status': 'warmed_up',
            'sentence_transformer': model is not None,
            'spacy': analyzer.nlp is not None,
            'message': 'Models loaded and ready'
        })
        
    except Exception as e:
        print(f"Warmup error: {e}")
        return jsonify({
            'status': 'warmup_failed',
            'error': str(e)
        }), 500

@app.route('/bypass')
@cross_origin()
def serve_bypass():
    """Serve the ngrok bypass page"""
    return send_from_directory('..', 'bypass.html')

@app.route('/api/current-url')
@cross_origin()
def get_current_url():
    """Get the current ngrok URL if available"""
    try:
        # Try to read from the ngrok URL file
        import os
        url_file = os.path.join('..', 'ngrok_url.txt')
        if os.path.exists(url_file):
            with open(url_file, 'r') as f:
                ngrok_url = f.read().strip()
                if ngrok_url:
                    return jsonify({
                        'public_url': ngrok_url,
                        'local_url': 'http://localhost:5000',
                        'bypass_url': f"{ngrok_url}?ngrok-skip-browser-warning=true"
                    })
        
        # Fallback to localhost
        return jsonify({
            'public_url': 'http://localhost:5000',
            'local_url': 'http://localhost:5000',
            'bypass_url': 'http://localhost:5000'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production