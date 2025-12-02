"""
Query processing module
Handles boolean and ranked query processing with multiple strategies
"""

import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from .core import IndexInfo, Optimizations, QueryProc
from .index import InvertedIndex


class BooleanExprParser:
    """
    Enhanced recursive descent parser for boolean expressions
    Respects operator precedence: PHRASE > NOT > AND > OR
    Supports phrase queries: PHRASE "word1 word2 word3"
    """
    
    def __init__(self, index: 'InvertedIndex'):
        self.index = index
        self.tokens = []
        self.pos = 0
    
    def parse(self, query: str) -> Set[int]:
        """Parse and evaluate boolean query"""
        self.tokens = self._tokenize(query)
        self.pos = 0
        
        if not self.tokens:
            return set()
        
        result = self._parse_or_expr()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        
        return result
    
    def _tokenize(self, query: str) -> List[Tuple[str, str]]:
        """
        Tokenize query into terms and operators
        """
        tokens = []
        pattern = r'PHRASE\s+"([^"]+)"|"([^"]+)"|\bAND\b|\bOR\b|\bNOT\b|\(|\)'
        
        for match in re.finditer(pattern, query, re.IGNORECASE):
            matched_text = match.group(0)
            
            if match.group(1):  # PHRASE "..." captured
                phrase_content = match.group(1)
                tokens.append(('PHRASE', phrase_content))
            
            elif match.group(2):  # Regular "..." captured
                term_content = match.group(2)
                tokens.append(('TERM', term_content))
            
            elif matched_text.upper() == 'AND':
                tokens.append(('AND', 'AND'))
            
            elif matched_text.upper() == 'OR':
                tokens.append(('OR', 'OR'))
            
            elif matched_text.upper() == 'NOT':
                tokens.append(('NOT', 'NOT'))
            
            elif matched_text == '(':
                tokens.append(('LPAREN', '('))
            
            elif matched_text == ')':
                tokens.append(('RPAREN', ')'))
        
        return tokens
    
    def _current_token(self):
        """Get current token without consuming"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def _consume(self, expected_type=None):
        """Consume and return current token"""
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token
    
    def _parse_or_expr(self):
        """OR expressions (lowest precedence)"""
        left = self._parse_and_expr()
        
        while self._current_token() and self._current_token()[0] == 'OR':
            self._consume('OR')
            right = self._parse_and_expr()
            left = left.union(right)
        
        return left
    
    def _parse_and_expr(self):
        """AND expressions (middle precedence)"""
        left = self._parse_not_expr()
        
        while self._current_token() and self._current_token()[0] == 'AND':
            self._consume('AND')
            right = self._parse_not_expr()
            left = left.intersection(right)
        
        return left
    
    def _parse_not_expr(self):
        """NOT expressions (high precedence)"""
        if self._current_token() and self._current_token()[0] == 'NOT':
            self._consume('NOT')
            operand = self._parse_not_expr()
            all_docs = set(range(self.index.num_docs))
            return all_docs.difference(operand)
        
        return self._parse_phrase_expr()
    
    def _parse_phrase_expr(self):
        """PHRASE expressions (highest precedence)"""
        token = self._current_token()
        
        if token and token[0] == 'PHRASE':
            phrase_token = self._consume('PHRASE')
            phrase_text = phrase_token[1]
            return self._evaluate_phrase(phrase_text)
        
        return self._parse_primary()
    
    def _evaluate_phrase(self, phrase_text: str) -> Set[int]:
        """Evaluate phrase query: "word1 word2 word3" """
        phrase_words = phrase_text.split()
        
        if not phrase_words:
            return set()
        
        processed_terms = []
        for word in phrase_words:
            processed = self.index.preprocessor.preprocess(word)
            if processed:
                processed_terms.append(processed[0])
            else:
                return set()
        
        if not processed_terms:
            return set()
        
        first_term = processed_terms[0]
        term_info = self.index.get_term_info(first_term)
        
        if not term_info or 'docs' not in term_info:
            return set()
        
        candidate_docs = set(term_info['docs'])
        
        if len(processed_terms) == 1:
            return candidate_docs
        
        result_docs = set()
        
        for doc_id in candidate_docs:
            if self._check_phrase_positions(doc_id, processed_terms):
                result_docs.add(doc_id)
        
        return result_docs
    
    def _check_phrase_positions(self, doc_id: int, processed_terms: List[str]) -> bool:
        """Check if phrase terms appear consecutively in document"""
        first_term = processed_terms[0]
        term_info = self.index.get_term_info(first_term)
        
        if not term_info or 'positions' not in term_info:
            return False
        
        positions_dict = term_info['positions']
        if doc_id not in positions_dict:
            return False
        
        first_positions = positions_dict[doc_id]
        
        for start_pos in first_positions:
            phrase_found = True
            current_pos = start_pos
            
            for term in processed_terms[1:]:
                expected_pos = current_pos + 1
                term_info = self.index.get_term_info(term)
                
                if not term_info or 'positions' not in term_info:
                    phrase_found = False
                    break
                
                positions_dict = term_info['positions']
                
                if doc_id not in positions_dict:
                    phrase_found = False
                    break
                
                term_positions = positions_dict[doc_id]
                
                if expected_pos in term_positions:
                    current_pos = expected_pos
                else:
                    phrase_found = False
                    break
            
            if phrase_found:
                return True
        
        return False
    
    def _parse_primary(self):
        """PRIMARY := TERM | (EXPR)"""
        token = self._current_token()
        
        if not token:
            raise ValueError("Expected term or '('")
        
        if token[0] == 'LPAREN':
            self._consume('LPAREN')
            result = self._parse_or_expr()
            self._consume('RPAREN')
            return result
        
        elif token[0] == 'TERM':
            term_token = self._consume('TERM')
            term_text = term_token[1]
            processed = self.index.preprocessor.preprocess(term_text)
            if not processed:
                return set()
            return set(self.index.get_postings(processed[0]))
        
        else:
            raise ValueError(f"Unexpected token: {token[0]}")


class QueryProcessor:
    """
    Handles boolean and ranked query processing
    Supports Term-at-a-time and Document-at-a-time strategies
    """
    
    def __init__(self, index: InvertedIndex, 
                 strategy: QueryProc = QueryProc.TERMatat,
                 optimization: Optimizations = Optimizations.Null):
        self.index = index
        self.strategy = strategy
        self.optimization = optimization
        self.parser = BooleanExprParser(index)
    
    def process_boolean_query(self, query: str) -> Set[int]:
        """Process boolean query with AND, OR, NOT, PHRASE operators"""
        try:
            parser = BooleanExprParser(self.index)
            result_docids = parser.parse(query)
            return result_docids
        except Exception as e:
            print(f"Query parsing error: {e}")
            return set()
        
    def process_ranked_query(self, query: str, top_k: int = 10):
        """Process ranked query and return top-k documents with scores"""
        if self.strategy == QueryProc.TERMatat:
            results = self._term_at_a_time(query, top_k)
        else:
            results = self._document_at_a_time(query, top_k)
        
        # Enrich results with titles from metadata
        show_results = []
        for doc_id, score in results:
            metadata = self.index.doc_metadata.get(doc_id, {})
            res = {
                "doc_id": str(doc_id),
                "title": metadata.get("title", "Untitled Document"),
                "author": metadata.get("author", "Unknown"),
                "published": metadata.get("published", "N/A"),
                "score": float(score)
            }
            show_results.append(res)

        return show_results

    def _term_at_a_time(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Term-at-a-time query processing with optional Skipping optimization"""
        query_terms = self.index.preprocessor.preprocess(query)
        scores = defaultdict(float)
        
        if self.optimization == Optimizations.Skipping:
            return self._term_at_a_time_with_skipping(query_terms, top_k)
        
        for term in query_terms:
            term_info = self.index.get_term_info(term)
            if not term_info:
                continue
            
            if self.index.index_info == IndexInfo.TFIDF:
                for doc_id, tfidf_score in term_info.get('tfidf', {}).items():
                    scores[doc_id] += tfidf_score
            elif self.index.index_info == IndexInfo.WORDCOUNT:
                for doc_id, tf in term_info.get('tf', {}).items():
                    scores[doc_id] += tf
            else:
                for doc_id in term_info.get('docs', []):
                    scores[doc_id] += 1
        
        if self.optimization == Optimizations.Thresholding:
            threshold = self._calculate_threshold(scores, top_k)
            scores = {doc_id: score for doc_id, score in scores.items() 
                     if score >= threshold}
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if self.optimization == Optimizations.EarlyStopping:
            return ranked[:top_k]
        
        return ranked[:top_k]
    
    def _term_at_a_time_with_skipping(self, query_terms: List[str], 
                                      top_k: int) -> List[Tuple[int, float]]:
        """Term-at-a-time with Skipping optimization"""
        if not query_terms:
            return []
        
        term_data = []
        for term in query_terms:
            term_info = self.index.get_term_info(term)
            if term_info:
                df = term_info.get('df', 0)
                term_data.append((term, term_info, df))
        
        if not term_data:
            return []
        
        term_data.sort(key=lambda x: x[2])
        rarest_term, rarest_info, _ = term_data[0]
        candidate_docs = set(rarest_info.get('docs', []))
        scores = defaultdict(float)
        
        for term, term_info, df in term_data:
            if self.index.index_info == IndexInfo.TFIDF:
                tfidf_scores = term_info.get('tfidf', {})
                for doc_id in candidate_docs:
                    if doc_id in tfidf_scores:
                        scores[doc_id] += tfidf_scores[doc_id]
            
            elif self.index.index_info == IndexInfo.WORDCOUNT:
                tf_scores = term_info.get('tf', {})
                for doc_id in candidate_docs:
                    if doc_id in tf_scores:
                        scores[doc_id] += tf_scores[doc_id]
            
            else:  # Boolean
                term_docs = set(term_info.get('docs', []))
                for doc_id in candidate_docs:
                    if doc_id in term_docs:
                        scores[doc_id] += 1
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def _document_at_a_time(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Document-at-a-time query processing"""
        query_terms = self.index.preprocessor.preprocess(query)
        candidate_docs = set()
        term_infos = []
        
        for term in query_terms:
            term_info = self.index.get_term_info(term)
            if term_info:
                candidate_docs.update(term_info.get('docs', []))
                term_infos.append((term, term_info))
        
        if self.optimization == Optimizations.Skipping and len(term_infos) > 1:
            term_infos_sorted = sorted(term_infos, key=lambda x: x[1].get('df', 0))
            rarest_docs = set(term_infos_sorted[0][1].get('docs', []))
            candidate_docs = candidate_docs.intersection(rarest_docs)
        
        scores = {}
        for doc_id in candidate_docs:
            score = 0
            for term, term_info in term_infos:
                if self.index.index_info == IndexInfo.TFIDF:
                    score += term_info.get('tfidf', {}).get(doc_id, 0)
                elif self.index.index_info == IndexInfo.WORDCOUNT:
                    score += term_info.get('tf', {}).get(doc_id, 0)
                else:
                    score += 1 if doc_id in term_info.get('docs', []) else 0
            scores[doc_id] = score
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def _calculate_threshold(self, scores: Dict[int, float], top_k: int) -> float:
        """Calculate score threshold for optimization"""
        if len(scores) <= top_k:
            return 0
        sorted_scores = sorted(scores.values(), reverse=True)
        return sorted_scores[top_k] if len(sorted_scores) > top_k else 0
