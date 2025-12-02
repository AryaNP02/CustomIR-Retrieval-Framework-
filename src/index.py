"""
Inverted Index implementation
Supports multiple storage backends and compression methods
"""

import json
import pickle
import sqlite3
import redis
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional
import math

from .core import IndexInfo, DataStore, Compression, CompressionUtils
from .preprocessor import TextPreprocessor


class InvertedIndex:
    """
    Core inverted index structure
    Supports different index types, storage backends, and compression
    """
    
    def __init__(self, 
                 index_info: IndexInfo = IndexInfo.BOOLEAN,
                 datastore: DataStore = DataStore.CUSTOM,
                 compression: Compression = Compression.NONE,
                 index_name: str = "default"):
        
        self.index_info = index_info
        self.datastore = datastore
        self.compression = compression
        self.index_name = index_name
        self.preprocessor = TextPreprocessor()
        
        # ==========  UUID to Integer Mapping ==========
        self.uuid_to_docid = {}  # Maps UUID string -> integer doc_id
        self.docid_to_uuid = {}  # Maps integer doc_id -> UUID string
        self.next_doc_id = 0     # Counter for generating unique integer IDs
        # ================================================
        
        # In-memory structures
        self.index: Dict[str, Dict] = defaultdict(lambda: {
            'docs': [],
            'positions': defaultdict(list),
            'tf': defaultdict(int),
            'df': 0
        })
        self.doc_lengths: Dict[int, int] = {}
        self.doc_metadata: Dict[int, Dict] = {}
        self.num_docs = 0
        self.avg_doc_length = 0
        
        # Storage backend
        self.db_connection = None
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage backend"""
        if hasattr(self, "db_connection") and self.db_connection:
            try:
                self.db_connection.close()
                print("Closed existing DB connection.")
            except Exception:
                pass
            finally:
                self.db_connection = None

        if self.datastore == DataStore.DB1:
            # SQLite
            db_path = f"indices/{self.index_name}.db"
            Path("indices").mkdir(exist_ok=True)
            self.db_connection = sqlite3.connect(db_path)
            self._create_sqlite_schema()
        elif self.datastore == DataStore.DB2:
            try:
                # Try connecting first
                self.db_connection = redis.Redis(host='localhost', port=6379, db=0)
                self.db_connection.ping()
            except redis.exceptions.ConnectionError:
                import subprocess
                import time
                try:
                    # Start Redis in the background
                    subprocess.Popen(["redis-server", "--daemonize", "yes"])
                    time.sleep(1)  # Give it a moment to start
                    self.db_connection = redis.Redis(host='localhost', port=6379, db=0)
                    self.db_connection.ping()
                except Exception as e:
                    print(f"[Redis] Failed to start: {e}")
                    print("Using CUSTOM storage")
                    self.datastore = DataStore.CUSTOM
                    self.db_connection = None

    def _get_internal_doc_id(self, uuid: str) -> int:
        """
        Convert UUID string to internal integer document ID.
        Creates new ID if UUID not seen before.
        
        Args:
            uuid: String UUID like "3585e6ee69707c5862f374b499698f49c38bd1d6"
        
        Returns:
            Integer document ID (0, 1, 2, ...)
        """
        if uuid not in self.uuid_to_docid:
            doc_id = self.next_doc_id
            self.uuid_to_docid[uuid] = doc_id
            self.docid_to_uuid[doc_id] = uuid
            self.next_doc_id += 1
            return doc_id
        
        return self.uuid_to_docid[uuid]

    def _get_uuid_from_docid(self, doc_id: int) -> str:
        """Get original UUID string from internal integer doc_id"""
        return self.docid_to_uuid.get(doc_id, str(doc_id))

    def _create_sqlite_schema(self):
        """Create SQLite schema for index storage"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT PRIMARY KEY,
                data BLOB,
                df INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                length REAL,
                metadata TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        self.db_connection.commit()
        
    
    def add_document(self, uuid: str, text: str, metadata: Dict = None):
        """
        Add a document to the index using UUID as identifier.

        Args:
            uuid: Document UUID string (e.g., "3585e6ee69707c5862f374b499698f49c38bd1d6")
            text: Document text content
            metadata: Optional metadata dictionary
        """
        # Convert UUID to internal integer document ID
        doc_id = self._get_internal_doc_id(uuid)

        # Preprocess with positions
        terms_with_positions = self.preprocessor.preprocess_with_positions(text)

        # Update document length
        self.doc_lengths[doc_id] = len(terms_with_positions)

        # Ensure metadata includes UUID
        if metadata is None:
            metadata = {}
        metadata['uuid'] = uuid
        self.doc_metadata[doc_id] = metadata

        # Build term → positions map
        term_positions = defaultdict(list)
        for term, position in terms_with_positions:
            term_positions[term].append(position)

        # Compute TF only if needed (for WORDCOUNT or TFIDF)
        term_counts = None
        if self.index_info in {IndexInfo.WORDCOUNT, IndexInfo.TFIDF}:
            term_counts = Counter(term for term, _ in terms_with_positions)

        # Update inverted index
        for term, positions in term_positions.items():
            if doc_id not in self.index[term]['docs']:
                self.index[term]['docs'].append(doc_id)
                self.index[term]['df'] += 1

            # Store positions for all index types
            self.index[term]['positions'][doc_id] = positions

            # Add extra info only for WORDCOUNT or TFIDF
            if self.index_info in {IndexInfo.WORDCOUNT, IndexInfo.TFIDF} and term_counts:
                self.index[term]['tf'][doc_id] = term_counts[term]

        # Increment total document count
        self.num_docs += 1
    
    def finalize_index(self):
        """
        Finalize index: sort postings, calculate statistics, compute TF-IDF
        """
        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        # Sort postings lists and calculate TF-IDF if needed
        for term in self.index:
            self.index[term]['docs'].sort()
            
            if self.index_info == IndexInfo.TFIDF:
                # Calculate IDF
                df = self.index[term]['df']
                idf = math.log((self.num_docs + 1) / (df + 1))
                
                # Calculate TF-IDF for each document
                self.index[term]['tfidf'] = {}
                for doc_id in self.index[term]['docs']:
                    tf = self.index[term]['tf'][doc_id]
                    # Normalize by document length
                    normalized_tf = tf / self.doc_lengths.get(doc_id, 1)
                    tfidf = normalized_tf * idf
                    self.index[term]['tfidf'][doc_id] = tfidf
    
    def save_index(self):
        """Save index to disk using configured storage backend"""
        if self.datastore == DataStore.CUSTOM:
            self._save_custom()
        elif self.datastore == DataStore.DB1:
            self._save_sqlite()
        elif self.datastore == DataStore.DB2:
            self._save_redis()
    
    def _save_custom(self):
        """Save using custom pickle format with UUID mapping"""
        Path("indices").mkdir(exist_ok=True)
        index_path = f"indices/{self.index_name}_custom.pkl"
        
        compressed_index = {}
        for term, data in self.index.items():
            compressed_index[term] = {
                'docs_compressed': CompressionUtils.compress_postings(
                    data['docs'], self.compression
                ),
                'df': data['df'],
                'positions': dict(data['positions']),
                'tf': dict(data['tf']) if 'tf' in data else {},
                'tfidf': data.get('tfidf', {})
            }
        
        with open(index_path, 'wb') as f:
            pickle.dump({
                'index': compressed_index,
                'doc_lengths': self.doc_lengths,
                'doc_metadata': self.doc_metadata,
                'uuid_to_docid': self.uuid_to_docid,
                'docid_to_uuid': self.docid_to_uuid,
                'num_docs': self.num_docs,
                'avg_doc_length': self.avg_doc_length,
                'config': {
                    'index_info': self.index_info,
                    'compression': self.compression
                }
            }, f)
    
    def _save_sqlite(self):
        """Save using SQLite with UUID mapping"""
        cursor = self.db_connection.cursor()
        
        # Save UUID mapping in separate table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uuid_mapping (
                doc_id INTEGER PRIMARY KEY,
                uuid TEXT UNIQUE NOT NULL
            )
        ''')
        
        # Insert UUID mappings
        for doc_id, uuid in self.docid_to_uuid.items():
            try:
                cursor.execute(
                    'INSERT OR REPLACE INTO uuid_mapping (doc_id, uuid) VALUES (?, ?)',
                    (int(doc_id), str(uuid))
                )
            except Exception as e:
                print(f"Error saving UUID mapping {doc_id}->{uuid}: {e}")
        
        # Save postings (already using integer doc_ids)
        for term, data in self.index.items():
            compressed_data = pickle.dumps({
                'docs': data['docs'],
                'positions': dict(data['positions']),
                'tf': dict(data.get('tf', {})),
                'tfidf': data.get('tfidf', {})
            })

            try:
                cursor.execute(
                    'INSERT OR REPLACE INTO postings (term, data, df) VALUES (?, ?, ?)',
                    (str(term), sqlite3.Binary(compressed_data), int(data['df']))
                )
            except sqlite3.IntegrityError as e:
                print(f"[postings] {term}: {e}")
        
        # Save documents with UUID reference
        for doc_id, length in self.doc_lengths.items():
            uuid = self.docid_to_uuid.get(doc_id, str(doc_id))
            metadata_json = json.dumps(self.doc_metadata.get(doc_id, {}))
            try:
                cursor.execute(
                    'INSERT OR REPLACE INTO documents (doc_id, length, metadata) VALUES (?, ?, ?)',
                    (str(doc_id), float(length), metadata_json)
                )
            except Exception as e:
                print(f"[documents] {doc_id}: {e}")
        
        self.db_connection.commit()

    
    def _save_redis(self):
        """Save using Redis with UUID mapping"""
        if not self.db_connection:
            return
        
        prefix = f"{self.index_name}:"
        
        # Save UUID mapping
        for doc_id, uuid in self.docid_to_uuid.items():
            key = f"{prefix}uuid:{doc_id}"
            self.db_connection.set(key, uuid)
        
        # Save postings
        for term, data in self.index.items():
            key = f"{prefix}term:{term}"
            value = pickle.dumps({
                'docs': data['docs'],
                'positions': dict(data['positions']),
                'tf': dict(data['tf']) if 'tf' in data else {},
                'tfidf': data.get('tfidf', {}),
                'df': data['df']
            })
            self.db_connection.set(key, value)
        
        # Save metadata
        self.db_connection.set(f"{prefix}uuid_mapping", pickle.dumps(self.uuid_to_docid))
        self.db_connection.set(f"{prefix}docid_mapping", pickle.dumps(self.docid_to_uuid))
        self.db_connection.set(f"{prefix}num_docs", self.num_docs)
        self.db_connection.set(f"{prefix}avg_doc_length", self.avg_doc_length)
        self.db_connection.set(f"{prefix}doc_lengths", pickle.dumps(self.doc_lengths))
        self.db_connection.set(f"{prefix}doc_metadata", pickle.dumps(self.doc_metadata))
    
    def load_index(self):
        """Load index from disk"""
        if self.datastore == DataStore.CUSTOM:
            self._load_custom()
        elif self.datastore == DataStore.DB1:
            self._load_sqlite()
        elif self.datastore == DataStore.DB2:
            self._load_redis()
    
    def _load_custom(self):
        """Load from custom pickle format with UUID mapping"""
        index_path = f"indices/{self.index_name}_custom.pkl"
        if not Path(index_path).exists():
            return
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore UUID mappings
        self.uuid_to_docid = data.get('uuid_to_docid', {})
        self.docid_to_uuid = data.get('docid_to_uuid', {})
        self.next_doc_id = max(self.docid_to_uuid.keys()) + 1 if self.docid_to_uuid else 0
        
        # Decompress postings
        for term, compressed_data in data['index'].items():
            self.index[term] = {
                'docs': CompressionUtils.decompress_postings(
                    compressed_data['docs_compressed'], 
                    data['config']['compression']
                ),
                'df': compressed_data['df'],
                'positions': defaultdict(list, compressed_data['positions']),
                'tf': defaultdict(int, compressed_data.get('tf', {})),
                'tfidf': compressed_data.get('tfidf', {})
            }
        
        self.doc_lengths = data['doc_lengths']
        self.doc_metadata = data['doc_metadata']
        self.num_docs = data['num_docs']
        self.avg_doc_length = data['avg_doc_length']
    
    def _load_sqlite(self):
        """Load from SQLite"""
        cursor = self.db_connection.cursor()
        
        # Load postings
        cursor.execute('SELECT term, data, df FROM postings')
        for term, data_blob, df in cursor.fetchall():
            data = pickle.loads(data_blob)
            self.index[term] = {
                'docs': data['docs'],
                'df': df,
                'positions': defaultdict(list, data['positions']),
                'tf': defaultdict(int, data.get('tf', {})),
                'tfidf': data.get('tfidf', {})
            }
        
        # Load documents
        cursor.execute('SELECT doc_id, length, metadata FROM documents')
        for doc_id, length, metadata_json in cursor.fetchall():
            self.doc_lengths[doc_id] = length
            self.doc_metadata[doc_id] = json.loads(metadata_json)
        
        # Load statistics
        cursor.execute('SELECT value FROM index_stats WHERE key = ?', ('num_docs',))
        result = cursor.fetchone()
        if result:
            self.num_docs = int(result[0])
        
        cursor.execute('SELECT value FROM index_stats WHERE key = ?', ('avg_doc_length',))
        result = cursor.fetchone()
        if result:
            self.avg_doc_length = float(result[0])
    
    def _load_redis(self):
        """Load from Redis"""
        if not self.db_connection:
            return
        
        prefix = f"{self.index_name}:"
        
        # Load metadata first
        num_docs = self.db_connection.get(f"{prefix}num_docs")
        if num_docs:
            self.num_docs = int(num_docs)
        
        avg_doc_length = self.db_connection.get(f"{prefix}avg_doc_length")
        if avg_doc_length:
            self.avg_doc_length = float(avg_doc_length)
        
        doc_lengths = self.db_connection.get(f"{prefix}doc_lengths")
        if doc_lengths:
            self.doc_lengths = pickle.loads(doc_lengths)
        
        doc_metadata = self.db_connection.get(f"{prefix}doc_metadata")
        if doc_metadata:
            self.doc_metadata = pickle.loads(doc_metadata)
        
        # Load postings
        for key in self.db_connection.scan_iter(f"{prefix}term:*"):
            term = key.decode().split(':')[-1]
            data = pickle.loads(self.db_connection.get(key))
            self.index[term] = {
                'docs': data['docs'],
                'df': data['df'],
                'positions': defaultdict(list, data['positions']),
                'tf': defaultdict(int, data.get('tf', {})),
                'tfidf': data.get('tfidf', {})
            }
    
    def get_postings(self, term: str) -> List[int]:
        """Get postings list for a term"""
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return []
        term = processed_terms[0]
        
        if term in self.index:
            return self.index[term]['docs']
        return []
    
    def get_term_info(self, term: str) -> Dict:
        """Get all information about a term"""
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return {}
        term = processed_terms[0]
        
        return self.index.get(term, {})
