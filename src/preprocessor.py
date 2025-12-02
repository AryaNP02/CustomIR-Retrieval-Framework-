"""
Text preprocessing module
Handles tokenization, stemming, and stop word removal
"""

import re
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Handles text preprocessing: tokenization, stemming, stop word removal"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text: lowercase, tokenize, remove stopwords, stem
        Returns list of processed terms
        """
        if not text:
            return []
        
        # Lowercase and tokenize
        text = text.lower()
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if nltk fails
            tokens = text.split()
        
        # Remove stop words and stem
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)
        
        return processed_tokens
    
    def preprocess_with_positions(self, text: str) -> List[Tuple[str, int]]:
        """
        Preprocess text and return terms with their positions
        Returns list of (term, position) tuples
        """
        if not text:
            return []
        
        text = text.lower()
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        result = []
        position = 0
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                stemmed = self.stemmer.stem(token)
                result.append((stemmed, position))
                position += 1
        
        return result
