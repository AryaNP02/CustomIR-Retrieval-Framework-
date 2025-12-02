"""
Index builder and query generator module
Handles index building from data and test query generation
"""

import json
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List
import numpy as np

from .core import IndexInfo, DataStore, Compression, QueryProc, Optimizations
from .index import InvertedIndex
from .query_processor import QueryProcessor


class IndexBuilder:
    """
    Main interface for building indices with different configurations
    """
    
    def __init__(self, version_string: str):
        """
        Initialize with version string: SelfIndex-v1.xyziq
        x = IndexInfo (1=BOOLEAN, 2=WORDCOUNT, 3=TFIDF)
        y = DataStore (1=CUSTOM, 2=DB1, 3=DB2)
        z = Compression (1=NONE, 2=CODE, 3=CLIB)
        i = Optimizations (0=Null, sp=Skipping, th=Thresholding, es=EarlyStopping)
        q = QueryProc (T=TERMatat, D=DOCatat)
        """
        self.version = version_string
        self.config = self._parse_version(version_string)
        self.index = None
    
    def _parse_version(self, version: str) -> Dict:
        """Parse version string into configuration"""
        # Extract xyziq from SelfIndex-v1.xyziq
        match = re.search(r'v1\.(\d)(\d)(\d)([a-z0-9]+)([TD])', version)
        if not match:
            raise ValueError(f"Invalid version string: {version}")
        
        x, y, z, i, q = match.groups()
        
        return {
            'index_info': IndexInfo(int(x)),
            'datastore': DataStore(int(y)),
            'compression': Compression(int(z)),
            'optimization': Optimizations(i),
            'query_proc': QueryProc(q)
        }
    
    def build_index(self, data_dir: Path, max_docs: int = None):
        """Build index from data directory"""
        print(f"Building index with configuration:")
        print(f"  IndexInfo: {self.config['index_info']}")
        print(f"  DataStore: {self.config['datastore']}")
        print(f"  Compression: {self.config['compression']}")
        print(f"  Optimization: {self.config['optimization']}")
        print(f"  QueryProc: {self.config['query_proc']}")
        
        # Create index
        self.index = InvertedIndex(
            index_info=self.config['index_info'],
            datastore=self.config['datastore'],
            compression=self.config['compression'],
            index_name=self.version
        )
        
        # Find all JSON files
        start_time = time.time()
        doc_count = 0

        all_files = list(data_dir.rglob('*'))
        print(f"\nFound {len(all_files)} files (recursively scanning subdirectories)")

        for path in sorted(all_files):
            if max_docs and doc_count >= max_docs:
                break

            # --- Case 1: JSON file ---
            if path.is_file() and path.suffix.lower() == '.json':
                try:
                    with open(path, 'rb') as fh:
                        raw = fh.read()
                        try:
                            s = raw.decode('utf-8')
                        except Exception:
                            s = raw.decode('latin-1')
                        data = json.loads(s)

                        doc_uuid = (
                            data.get('uuid')
                            or data.get('thread', {}).get('uuid')
                            or f"{path.parent.name}/{path.name}"
                        )
                        
                        # Ensure UUID is a string
                        doc_uuid = str(doc_uuid).strip()
                        
                        # Validate UUID is not empty
                        if not doc_uuid:
                            print(f"⚠️ Skipping {path}: Empty UUID")
                            continue

                        title = data.get('title', '')
                        text = data.get('text', '')
                        full_text = f"{title} {text}".strip()
                        if not full_text:
                            continue

                        metadata = {
                            'title': title,
                            'author': data.get('author'),
                            'published': data.get('published'),
                            'language': data.get('language'),
                            'url': data.get('url'),
                            'source_file': str(path)
                        }
                    
                        self.index.add_document(doc_uuid, full_text, metadata)
                        doc_count += 1

                        if doc_count % 100 == 0:
                            print(f"Indexed {doc_count} documents...", end='\r')

                except Exception as e:
                    print(f"⚠️ Failed reading {path}: {e}")

            # --- Case 2: ZIP archive ---
            elif path.is_file() and path.suffix.lower() == '.zip':
                try:
                    with zipfile.ZipFile(path, 'r') as zf:
                        for name in zf.namelist():
                            if not name.lower().endswith('.json'):
                                continue
                            if max_docs and doc_count >= max_docs:
                                break
                            try:
                                with zf.open(name) as fh:
                                    raw = fh.read()
                                    try:
                                        s = raw.decode('utf-8')
                                    except Exception:
                                        s = raw.decode('latin-1')
                                    data = json.loads(s)

                                    doc_uuid = (
                                        data.get('uuid')
                                        or data.get('thread', {}).get('uuid')
                                        or f"{path.stem}/{name}"
                                    )
                                    
                                    doc_uuid = str(doc_uuid).strip()
                                    
                                    if not doc_uuid:
                                        print(f"⚠️ Skipping {name} in {path}: Empty UUID")
                                        continue

                                    title = data.get('title', '')
                                    text = data.get('text', '')
                                    full_text = f"{title} {text}".strip()
                                    if not full_text:
                                        continue

                                    metadata = {
                                        'title': title,
                                        'author': data.get('author'),
                                        'published': data.get('published'),
                                        'language': data.get('language'),
                                        'url': data.get('url'),
                                        'source_zip': str(path),
                                        'internal_name': name
                                    }

                                    self.index.add_document(doc_uuid, full_text, metadata)
                                    doc_count += 1
                                    
                                    if doc_count % 100 == 0:
                                        print(f"Indexed {doc_count} documents...", end='\r')

                            except Exception as e:
                                print(f"Failed reading {name} in {path}: {e}")
                except Exception as e:
                    print(f"Could not open ZIP {path}: {e}")

            else:
                continue

        print(f"\nIndexed {doc_count} documents in {time.time() - start_time:.2f}s")
        print("Finalizing and saving index...")

        self.index.finalize_index()
        self.index.save_index()
        print("Index build complete!")
        
        return self.index

    
    def load_index(self):
        """Load existing index"""
        self.index = InvertedIndex(
            index_info=self.config['index_info'],
            datastore=self.config['datastore'],
            compression=self.config['compression'],
            index_name=self.version
        )
        self.index.load_index()
        return self.index
    
    def get_query_processor(self) -> QueryProcessor:
        """Get query processor for this index"""
        if not self.index:
            raise ValueError("Index not built or loaded")
        
        return QueryProcessor(
            self.index,
            strategy=self.config['query_proc'],
            optimization=self.config['optimization']
        )


class TestQueryGenerator:
    """Generate valid boolean test queries including PHRASE operator"""
    
    @staticmethod
    def generate_queries(index: 'InvertedIndex', num_queries: int = 50) -> List[str]:
        """
        Generate VALID boolean test queries from index vocabulary following grammar
        
        Operator Precedence: PHRASE > NOT > AND > OR (highest to lowest)
        """
        queries = []
        
        # Get terms sorted by frequency
        term_freq = [(term, info['df']) for term, info in index.index.items()]
        term_freq.sort(key=lambda x: x[1], reverse=True)
        
        # Get common, medium, and rare terms
        total_terms = len(term_freq)
        common_terms = [t[0] for t in term_freq[:int(total_terms * 0.1)]]
        medium_terms = [t[0] for t in term_freq[int(total_terms * 0.3):int(total_terms * 0.6)]]
        rare_terms = [t[0] for t in term_freq[int(total_terms * 0.9):]]
        
        # Generate queries (14 patterns including PHRASE, cycling through)
        for i in range(num_queries):
            query_type = i % 14
            
            if query_type == 0:
                # Pattern: "word"
                if common_terms:
                    term = np.random.choice(common_terms)
                    queries.append(f'"{term}"')
            
            elif query_type == 1:
                # Pattern: "word1" AND "word2"
                if len(common_terms) >= 2:
                    t1, t2 = np.random.choice(common_terms, 2, replace=False)
                    queries.append(f'"{t1}" AND "{t2}"')
            
            elif query_type == 2:
                # Pattern: "word1" OR "word2"
                if len(common_terms) >= 2:
                    t1, t2 = np.random.choice(common_terms, 2, replace=False)
                    queries.append(f'"{t1}" OR "{t2}"')
            
            elif query_type == 3:
                # Pattern: NOT "word"
                if rare_terms:
                    term = np.random.choice(rare_terms)
                    queries.append(f'NOT "{term}"')
            
            elif query_type == 4:
                # Pattern: "word1" AND NOT "word2"
                if len(common_terms) >= 2:
                    t1 = np.random.choice(common_terms)
                    t2 = np.random.choice(rare_terms) if rare_terms else np.random.choice(common_terms)
                    queries.append(f'"{t1}" AND NOT "{t2}"')
            
            elif query_type == 5:
                # Pattern: "word1" AND "word2" AND "word3"
                if len(common_terms) >= 3:
                    t1, t2, t3 = np.random.choice(common_terms, 3, replace=False)
                    queries.append(f'"{t1}" AND "{t2}" AND "{t3}"')
            
            elif query_type == 6:
                # Pattern: ("word1" AND "word2") OR "word3"
                if len(common_terms) >= 3:
                    t1, t2, t3 = np.random.choice(common_terms, 3, replace=False)
                    queries.append(f'("{t1}" AND "{t2}") OR "{t3}"')
            
            elif query_type == 7:
                # Pattern: ("word1" OR "word2") AND NOT "word3"
                if len(common_terms) >= 3:
                    t1 = np.random.choice(common_terms)
                    t2 = np.random.choice(medium_terms) if medium_terms else np.random.choice(common_terms)
                    t3 = np.random.choice(rare_terms) if rare_terms else np.random.choice(common_terms)
                    queries.append(f'("{t1}" OR "{t2}") AND NOT "{t3}"')
            
            elif query_type == 8:
                # Pattern: ("word1" AND "word2") OR ("word3" AND NOT "word4")
                if len(common_terms) >= 4:
                    t1, t2, t3, t4 = np.random.choice(common_terms, 4, replace=False)
                    queries.append(f'("{t1}" AND "{t2}") OR ("{t3}" AND NOT "{t4}")')
            
            elif query_type == 9:
                # Pattern: "word1" AND ("word2" OR "word3" OR "word4")
                if len(common_terms) >= 4:
                    t1, t2, t3, t4 = np.random.choice(common_terms, 4, replace=False)
                    queries.append(f'"{t1}" AND ("{t2}" OR "{t3}" OR "{t4}")')
            
            elif query_type == 10:
                # Pattern: PHRASE "word1 word2"
                if len(common_terms) >= 2:
                    t1, t2 = np.random.choice(common_terms, 2, replace=False)
                    queries.append(f'PHRASE "{t1} {t2}"')
            
            elif query_type == 11:
                # Pattern: PHRASE "word1 word2 word3"
                if len(common_terms) >= 3:
                    t1, t2, t3 = np.random.choice(common_terms, 3, replace=False)
                    queries.append(f'PHRASE "{t1} {t2} {t3}"')
            
            elif query_type == 12:
                # Pattern: PHRASE "word1 word2" AND "word3"
                if len(common_terms) >= 3:
                    t1, t2, t3 = np.random.choice(common_terms, 3, replace=False)
                    queries.append(f'PHRASE "{t1} {t2}" AND "{t3}"')
            
            else:  # query_type == 13
                # Pattern: (PHRASE "word1 word2" OR PHRASE "word3 word4") AND NOT "word5"
                if len(common_terms) >= 5:
                    t1, t2, t3, t4, t5 = np.random.choice(common_terms, 5, replace=False)
                    queries.append(f'(PHRASE "{t1} {t2}" OR PHRASE "{t3} {t4}") AND NOT "{t5}"')
        
        return queries
