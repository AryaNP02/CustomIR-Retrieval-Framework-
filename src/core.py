"""
Core utilities and enums for SelfIndex
Handles compression, data types, and configuration enums
"""

import pickle
import gzip
from enum import Enum
from typing import List


class IndexInfo(Enum):
    """Type of information stored in the index"""
    BOOLEAN = 1
    WORDCOUNT = 2
    TFIDF = 3


class DataStore(Enum):
    """Storage backend for the index"""
    CUSTOM = 1
    DB1 = 2
    DB2 = 3


class Compression(Enum):
    """Compression method for postings lists"""
    NONE = 1
    CODE = 2
    CLIB = 3


class QueryProc(Enum):
    """Query processing strategy"""
    TERMatat = 'T'
    DOCatat = 'D'


class Optimizations(Enum):
    """Query optimizations"""
    Null = '0'
    Skipping = 'sp'
    Thresholding = 'th'
    EarlyStopping = 'es'


class CompressionUtils:
    """Handles compression of postings lists"""
    
    @staticmethod
    def variable_byte_encode(numbers: List[int]) -> bytes:
        """
        Variable byte encoding for gap-encoded integers
        Simple compression method
        """
        result = bytearray()
        for num in numbers:
            bytes_list = []
            while num >= 128:
                bytes_list.append(num % 128)
                num //= 128
            bytes_list.append(num + 128)  # Set continuation bit on last byte
            result.extend(bytes_list)
        return bytes(result)
    
    @staticmethod
    def variable_byte_decode(data: bytes) -> List[int]:
        """Decode variable byte encoded data"""
        numbers = []
        current = 0
        for byte in data:
            if byte < 128:
                current = current * 128 + byte
            else:
                current = current * 128 + (byte - 128)
                numbers.append(current)
                current = 0
        return numbers
    
    @staticmethod
    def gzip_compress(data: bytes) -> bytes:
        """Compress using gzip (library-based)"""
        return gzip.compress(data)
    
    @staticmethod
    def gzip_decompress(data: bytes) -> bytes:
        """Decompress gzip data"""
        return gzip.decompress(data)
    
    @staticmethod
    def compress_postings(postings: List[int], method: Compression) -> bytes:
        """
        Compress a list of integer document IDs.
        
        Args:
            postings: List of integer doc_ids (NOT strings or UUIDs)
            method: Compression method
        """
        if method == Compression.NONE:
            return pickle.dumps(postings)
        
        elif method == Compression.CODE:
            # Gap encoding + variable byte (requires integer postings)
            if not postings:
                return b''
            
            # Validate all postings are integers
            if not all(isinstance(p, int) for p in postings):
                raise TypeError(
                    f"Gap encoding requires integer postings. "
                    f"Got types: {set(type(p).__name__ for p in postings)}"
                )
            
            # Sort for consistent gap encoding
            postings_sorted = sorted(postings)
            
            # Calculate gaps
            gaps = [postings_sorted[0]] + [
                postings_sorted[i] - postings_sorted[i-1] 
                for i in range(1, len(postings_sorted))
            ]
            
            return CompressionUtils.variable_byte_encode(gaps)
        
        elif method == Compression.CLIB:
            # Pickle + gzip
            pickled = pickle.dumps(postings)
            return CompressionUtils.gzip_compress(pickled)
        
        return pickle.dumps(postings)

    @staticmethod
    def decompress_postings(data: bytes, method: Compression) -> List[int]:
        """Decompress a postings list to integer doc_ids"""
        if not data:
            return []
        
        if method == Compression.NONE:
            return pickle.loads(data)
        
        elif method == Compression.CODE:
            gaps = CompressionUtils.variable_byte_decode(data)
            if not gaps:
                return []
            
            # Reconstruct from gaps
            postings = [gaps[0]]
            for i in range(1, len(gaps)):
                postings.append(postings[-1] + gaps[i])
            
            return postings
        
        elif method == Compression.CLIB:
            pickled = CompressionUtils.gzip_decompress(data)
            return pickle.loads(pickled)
        
        return pickle.loads(data)
