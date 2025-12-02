"""
SelfIndex: A configurable inverted index implementation
Supports multiple index types, storage backends, and query processing strategies
"""

from .core import (
    IndexInfo, DataStore, Compression, QueryProc, Optimizations,
    CompressionUtils
)
from .preprocessor import TextPreprocessor
from .index import InvertedIndex
from .query_processor import BooleanExprParser, QueryProcessor
from .metrics import MetricsCollector, Reporter
from .index_builder import IndexBuilder, TestQueryGenerator

__version__ = "1.0.0"
__all__ = [
    "IndexInfo",
    "DataStore",
    "Compression",
    "QueryProc",
    "Optimizations",
    "CompressionUtils",
    "TextPreprocessor",
    "InvertedIndex",
    "BooleanExprParser",
    "QueryProcessor",
    "MetricsCollector",
    "Reporter",
    "IndexBuilder",
    "TestQueryGenerator",
]
