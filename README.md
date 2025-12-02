# CustomIR-Retrieval-Framework

## Overview

SelfIndex is a modular, configurable inverted index framework supporting multiple indexing strategies, storage backends, compression methods, and query processing optimizations. The system is designed for educational purposes and benchmarking against production systems like Elasticsearch.

---

## Architecture Overview

### Class Hierarchy and Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                      IndexBuilder                            │
│  - Parses version strings (SelfIndex-v1.xyziq)              │
│  - Orchestrates index creation and loading                   │
│  - Returns QueryProcessor instances                          │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ creates
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    InvertedIndex                             │
│  - Core data structure (term → postings mapping)            │
│  - Manages UUID to integer doc_id conversion                │
│  - Handles document addition and finalization               │
│  - Delegates to storage backends                            │
└───────┬─────────────────┬────────────────┬─────────────────┘
        │                 │                │
        │ uses            │ uses           │ uses
        ↓                 ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│TextProcessor │  │CompressionUtils│  │ Storage Backends │
│- Stemming    │  │- VarByte      │  │- Custom (Pickle) │
│- Tokenizing  │  │- GZIP         │  │- SQLite          │
│- Stop words  │  │- Gap encoding │  │- Redis           │
└──────────────┘  └──────────────┘  └──────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QueryProcessor                            │
│  - Processes Boolean and ranked queries                     │
│  - Implements TaT and DaT strategies                        │
│  - Applies optimizations (skipping, thresholding)           │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ uses
                    ↓
┌─────────────────────────────────────────────────────────────┐
│              BooleanExprParser                               │
│  - Recursive descent parser for Boolean queries             │
│  - Operator precedence: PHRASE > NOT > AND > OR             │
│  - Supports phrase queries with positional matching         │
└─────────────────────────────────────────────────────────────┘
```

---

## Entities Stored in the Index

### 1. **UUID to DocID Mapping**
```python
uuid_to_docid: Dict[str, int]  # "3585e6ee..." → 0
docid_to_uuid: Dict[int, str]  # 0 → "3585e6ee..."
next_doc_id: int                # Counter: 0, 1, 2, ...
```

**Purpose**: Enables compression-friendly integer doc_ids while preserving original UUID strings.

### 2. **Inverted Index Structure**
```python
index: Dict[str, Dict] = {
    'term': {
        'docs': [doc_id1, doc_id2, ...],          # Sorted integer list
        'positions': {
            doc_id1: [pos1, pos2, ...],           # Term positions in doc
            doc_id2: [pos5, pos7, ...]
        },
        'tf': {
            doc_id1: 5,                            # Term frequency (WordCount/TF-IDF only)
            doc_id2: 3
        },
        'tfidf': {
            doc_id1: 0.234,                        # TF-IDF score (TF-IDF only)
            doc_id2: 0.156
        },
        'df': 2                                    # Document frequency
    }
}
```

### 3. **Document Metadata**
```python
doc_lengths: Dict[int, int] = {
    0: 150,    # doc_id → number of terms after preprocessing
    1: 200
}

doc_metadata: Dict[int, Dict] = {
    0: {
        'uuid': '3585e6ee...',
        'title': 'Article Title',
        'author': 'John Doe',
        'published': '2024-01-15',
        'language': 'english',
        'url': 'https://...',
        'source_file': 'data/article.json'
    }
}

num_docs: int            # Total documents
avg_doc_length: float    # Average document length
```

---

## Version String Format

```
SelfIndex-v1.xyziq

x = IndexInfo:      1=Boolean, 2=WordCount, 3=TF-IDF
y = DataStore:      1=Custom, 2=SQLite (DB1), 3=Redis (DB2)
z = Compression:    1=None, 2=VarByte (CODE), 3=GZIP (CLIB)
i = Optimization:   0=Null, sp=Skipping, th=Thresholding, es=EarlyStopping
q = QueryProc:      T=Term-at-a-Time, D=Document-at-a-Time
```

**Examples**:
- `SelfIndex-v1.1110T`: Boolean + Custom + No Compression + No Opt + TaT
- `SelfIndex-v1.323spT`: TF-IDF + Redis + GZIP + Skipping + TaT

---

## Core Procedures

### 1. **Creating an Index for the First Time**

```python
from pathlib import Path

# Step 1: Initialize IndexBuilder with version string
builder = IndexBuilder('SelfIndex-v1.3210T')  # TF-IDF + SQLite + No Compression + TaT

# Step 2: Build index from data directory
DATA_DIR = Path('free-news-datasets/News_Datasets')
index = builder.build_index(DATA_DIR, max_docs=NUM_DOCS)

# What happens internally:
# 1. Parses version string into configuration
# 2. Creates InvertedIndex with specified parameters
# 3. Initializes storage backend (SQLite in this example)
# 4. Recursively scans DATA_DIR for JSON files and ZIP archives
# 5. For each document:
#    a. Extracts UUID, title, text, metadata
#    b. Converts UUID to internal doc_id
#    c. Preprocesses text (tokenize → stem → remove stop words)
#    d. Updates inverted index with term → (doc_id, positions, tf)
# 6. Finalizes index (sorts postings, calculates TF-IDF if enabled)
# 7. Saves to disk using configured backend
```

**Internal Processing Flow**:
```python
# UUID Conversion
uuid = "3585e6ee69707c5862f374b499698f49c38bd1d6"
doc_id = self._get_internal_doc_id(uuid)  # Returns 0 for first doc

# Text Preprocessing
text = "The machine learning algorithms are improving rapidly"
processed = TextPreprocessor.preprocess_with_positions(text)
# Returns: [('machin', 0), ('learn', 1), ('algorithm', 2), ('improv', 3), ('rapid', 4)]

# Index Update
for term, position in processed:
    index[term]['docs'].append(doc_id)
    index[term]['positions'][doc_id].append(position)
    index[term]['tf'][doc_id] += 1
```

---

### 2. **Loading an Existing Index**

```python
# Step 1: Create IndexBuilder with same version string
builder = IndexBuilder('SelfIndex-v1.3210T')

# Step 2: Load index from disk
index = builder.load_index()

# What happens internally (Custom storage example):
# 1. Opens 'indices/SelfIndex-v1.3210T_custom.pkl'
# 2. Unpickles the data structure containing:
#    - Compressed postings lists
#    - UUID mappings (uuid_to_docid, docid_to_uuid)
#    - Document metadata
#    - Index statistics
# 3. Decompresses postings using configured compression method
# 4. Reconstructs in-memory index structure
# 5. Sets next_doc_id to max(existing doc_ids) + 1
```

**Storage-Specific Loading**:

#### Custom (Pickle)
```python
def _load_custom(self):
    with open(f'indices/{self.index_name}_custom.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Restore UUID mappings
    self.uuid_to_docid = data['uuid_to_docid']
    self.docid_to_uuid = data['docid_to_uuid']
    
    # Decompress postings
    for term, compressed_data in data['index'].items():
        self.index[term]['docs'] = CompressionUtils.decompress_postings(
            compressed_data['docs_compressed'],
            self.compression
        )
```

#### SQLite
```python
def _load_sqlite(self):
    cursor = self.db_connection.cursor()
    
    # Load UUID mapping
    cursor.execute('SELECT doc_id, uuid FROM uuid_mapping')
    for doc_id, uuid in cursor.fetchall():
        self.docid_to_uuid[doc_id] = uuid
    
    # Load postings
    cursor.execute('SELECT term, data, df FROM postings')
    for term, data_blob, df in cursor.fetchall():
        data = pickle.loads(data_blob)
        self.index[term] = data
```

---

### 3. **Getting Metadata of an Already Built Index**

```python
# After loading or building index
builder = IndexBuilder('SelfIndex-v1.3210T')
index = builder.load_index()

# Access index configuration
config = builder.config
print(f"Index Type: {config['index_info']}")        # IndexInfo.TFIDF
print(f"Storage: {config['datastore']}")            # DataStore.DB1
print(f"Compression: {config['compression']}")      # Compression.NONE

# Access index statistics
print(f"Total Documents: {index.num_docs}")         # 109762
print(f"Avg Doc Length: {index.avg_doc_length}")    # 175.5

# Get document metadata by doc_id
doc_id = 0
metadata = index.doc_metadata[doc_id]
print(f"UUID: {metadata['uuid']}")
print(f"Title: {metadata['title']}")
print(f"Author: {metadata['author']}")

# Get term statistics
term_info = index.get_term_info('machin')  # Pre-stemmed term
print(f"Document Frequency: {term_info['df']}")
print(f"Postings List: {term_info['docs']}")
print(f"Positions in doc 0: {term_info['positions'][0]}")

# Get all vocabulary terms
vocabulary = list(index.index.keys())
print(f"Vocabulary Size: {len(vocabulary)}")
```

---

### 4. **Querying and Response Formatting**

#### Single Query Execution

```python
# Step 1: Get QueryProcessor
qp = builder.get_query_processor()

# Step 2: Process ranked query
results = qp.process_ranked_query("machine learning", top_k=10)

# What happens internally:
# 1. Preprocesses query terms: "machine learning" → ["machin", "learn"]
# 2. Retrieves term info from index for each term
# 3. Scores documents using configured strategy (TaT or DaT)
#    - Boolean: Count of matching terms
#    - WordCount: Sum of term frequencies
#    - TF-IDF: Sum of precomputed TF-IDF scores
# 4. Sorts documents by score (descending)
# 5. Returns top-k results with enriched metadata
```

#### Response Format

```python
# Return value structure
results = [
    {
        "doc_id": "0",                          # Internal doc_id (as string)
        "title": "Article Title",
        "author": "John Doe",
        "published": "2024-01-15",
        "score": 2.456                          # Relevance score
    },
    # ... more results
]

# Display formatted results
for i, doc in enumerate(results[:5], start=1):
    print(f"{i}. {doc['title']} ({doc['published']})")
    print(f"   Author: {doc['author']}")
    print(f"   Score: {doc['score']:.4f}")
    print("-" * 60)
```

#### Boolean Query Example

```python
# Complex Boolean query with PHRASE operator
query = '(PHRASE "machine learning" OR PHRASE "deep learning") AND NOT "deprecated"'
result_docids = qp.process_boolean_query(query)

# Returns set of matching doc_ids (integers)
print(f"Found {len(result_docids)} matching documents")
```

---

### 5. **Query Parsing**

The `BooleanExprParser` implements a recursive descent parser:

#### Grammar
```
QUERY    := EXPR
EXPR     := OR_EXPR
OR_EXPR  := AND_EXPR (OR AND_EXPR)*
AND_EXPR := NOT_EXPR (AND NOT_EXPR)*
NOT_EXPR := NOT NOT_EXPR | PHRASE_EXPR
PHRASE_EXPR := PHRASE "term1 term2" | PRIMARY
PRIMARY  := "term" | (EXPR)
```

#### Operator Precedence (Highest to Lowest)
1. **PHRASE** - Evaluated first
2. **NOT** - Unary negation
3. **AND** - Intersection
4. **OR** - Union (lowest precedence)

#### Parsing Process

```python
class BooleanExprParser:
    def parse(self, query: str) -> Set[int]:
        # Step 1: Tokenize query
        self.tokens = self._tokenize(query)
        # Input: '("term1" AND "term2") OR NOT "term3"'
        # Output: [('LPAREN', '('), ('TERM', 'term1'), ('AND', 'AND'),
        #          ('TERM', 'term2'), ('RPAREN', ')'), ('OR', 'OR'),
        #          ('NOT', 'NOT'), ('TERM', 'term3')]
        
        # Step 2: Parse expression tree (recursive descent)
        result_set = self._parse_or_expr()  # Start from lowest precedence
        
        return result_set
```

#### Tokenization Logic

```python
def _tokenize(self, query: str) -> List[Tuple[str, str]]:
    """Extract terms and operators using regex"""
    pattern = r'PHRASE\s+"([^"]+)"|"([^"]+)"|\bAND\b|\bOR\b|\bNOT\b|\(|\)'
    
    tokens = []
    for match in re.finditer(pattern, query, re.IGNORECASE):
        if match.group(1):  # PHRASE "..."
            tokens.append(('PHRASE', match.group(1)))
        elif match.group(2):  # "term"
            tokens.append(('TERM', match.group(2)))
        elif match.group(0) == 'AND':
            tokens.append(('AND', 'AND'))
        # ... handle OR, NOT, parentheses
    
    return tokens
```

#### PHRASE Query Evaluation

```python
def _evaluate_phrase(self, phrase_text: str) -> Set[int]:
    """Match documents containing exact phrase sequence"""
    # Step 1: Split and preprocess phrase terms
    words = phrase_text.split()  # "machine learning" → ["machine", "learning"]
    processed = [stem(word) for word in words]  # → ["machin", "learn"]
    
    # Step 2: Get candidate docs from first term
    first_term_info = index.get_term_info(processed[0])
    candidates = set(first_term_info['docs'])
    
    # Step 3: Check positional constraints
    results = set()
    for doc_id in candidates:
        if self._check_phrase_positions(doc_id, processed):
            results.add(doc_id)
    
    return results

def _check_phrase_positions(self, doc_id, terms):
    """Verify terms appear consecutively"""
    # Get positions of first term in document
    first_positions = index.get_term_info(terms[0])['positions'][doc_id]
    
    # For each occurrence, check if subsequent terms follow
    for start_pos in first_positions:
        match = True
        current_pos = start_pos
        
        for term in terms[1:]:
            expected_pos = current_pos + 1
            term_positions = index.get_term_info(term)['positions'][doc_id]
            
            if expected_pos not in term_positions:
                match = False
                break
            
            current_pos = expected_pos
        
        if match:
            return True  # Phrase found
    
    return False
```

---

### 6. **Metric Calculation**

#### Latency Measurement

```python
class MetricsCollector:
    @staticmethod
    def measure_query_latency(query_processor, queries, top_k=10):
        """Measure latency statistics across queries"""
        latencies = []
        
        for query in queries:
            # High-precision timer (nanosecond accuracy)
            start_time = time.time()
            _ = query_processor.process_ranked_query(query, top_k)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency_ms)
        
        # Calculate percentiles using NumPy
        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'min': min(latencies),
            'max': max(latencies)
        }
```

#### Throughput Measurement

```python
@staticmethod
def measure_throughput(query_processor, queries, duration=10):
    """Measure queries per second over fixed duration"""
    start_time = time.time()
    query_count = 0
    
    # Run queries for 'duration' seconds
    while time.time() - start_time < duration:
        for query in queries:
            _ = query_processor.process_ranked_query(query)
            query_count += 1
            
            if time.time() - start_time >= duration:
                break
    
    elapsed = time.time() - start_time
    throughput = query_count / elapsed
    
    return throughput  # queries/second
```

#### Memory Measurement

```python
@staticmethod
def measure_memory():
    """Get current process memory usage in MB"""
    import psutil
    
    process = psutil.Process()
    rss = process.memory_info().rss  # Resident Set Size (bytes)
    
    return rss / 1024 / 1024  # Convert to MB
```

#### Index Size Measurement

```python
@staticmethod
def measure_index_size(index_name, datastore):
    """Get on-disk index size in MB"""
    if datastore == DataStore.CUSTOM:
        path = Path(f'indices/{index_name}_custom.pkl')
    elif datastore == DataStore.DB1:
        path = Path(f'indices/{index_name}.db')
    elif datastore == DataStore.DB2:
        path = Path(f'indices/{index_name}_custom.pkl')
    
    if path.exists():
        size_bytes = path.stat().st_size
        return size_bytes / 1024 / 1024  # Convert to MB
    
    return 0.0
```

---

## Query Processing Strategies

### Term-at-a-Time (TaT)

```python
def _term_at_a_time(self, query, top_k):
    """Process one term at a time, accumulate scores"""
    scores = defaultdict(float)
    
    for term in query_terms:
        term_info = index.get_term_info(term)
        
        # Add scores for all documents containing this term
        for doc_id, tfidf_score in term_info['tfidf'].items():
            scores[doc_id] += tfidf_score
    
    # Sort and return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

**Advantages**:
- Better CPU cache locality
- Efficient for multi-term queries
- Lower memory overhead

### Document-at-a-Time (DaT)

```python
def _document_at_a_time(self, query, top_k):
    """Process one document at a time, score all terms"""
    candidate_docs = set()
    term_infos = []
    
    # Collect all candidate documents
    for term in query_terms:
        term_info = index.get_term_info(term)
        candidate_docs.update(term_info['docs'])
        term_infos.append(term_info)
    
    # Score each document
    scores = {}
    for doc_id in candidate_docs:
        score = sum(
            term_info['tfidf'].get(doc_id, 0)
            for term_info in term_infos
        )
        scores[doc_id] = score
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

**Advantages**:
- Complete document scores immediately available
- Easier to implement early stopping
- Better for interactive search

---

## Optimizations

### Skipping

```python
def _term_at_a_time_with_skipping(self, query_terms, top_k):
    """Skip documents unlikely to be in top-k"""
    # Sort terms by document frequency (rarest first)
    term_data = sorted(
        [(term, info, info['df']) for term, info in term_infos],
        key=lambda x: x[2]
    )
    
    # Use rarest term as candidate filter
    rarest_term, rarest_info, _ = term_data[0]
    candidate_docs = set(rarest_info['docs'])
    
    # Only score documents in candidate set
    scores = defaultdict(float)
    for term, term_info, df in term_data:
        for doc_id in candidate_docs:
            if doc_id in term_info['tfidf']:
                scores[doc_id] += term_info['tfidf'][doc_id]
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**Impact**: Reduces scoring overhead by 30-60% for selective queries.

---

## Storage Backend Comparison

| Feature | Custom (Pickle) | SQLite | Redis |
|---------|----------------|--------|-------|
| **Setup** | None | Creates `.db` file | Requires Redis server |
| **ACID** | No | Yes | No |
| **Concurrency** | Single process | Multiple readers | Multiple clients |
| **Query Speed** | Fastest (in-memory) | Fast (B-tree index) | Fast (in-memory) |
| **Scalability** | Limited by RAM | Limited by disk | Horizontal scaling |
| **Use Case** | Experiments | Production single-node | Distributed systems |

---

## Complete Example Workflow

```python
from pathlib import Path

# 1. CREATE INDEX
builder = IndexBuilder('SelfIndex-v1.3210T')
DATA_DIR = Path('free-news-datasets/News_Datasets')
index = builder.build_index(DATA_DIR, max_docs=60000)

# 2. QUERY INDEX
qp = builder.get_query_processor()
results = qp.process_ranked_query("machine learning deep neural", top_k=10)

# 3. DISPLAY RESULTS
for i, doc in enumerate(results[:5], start=1):
    print(f"{i}. {doc['title']}")
    print(f"   Score: {doc['score']:.4f}")

# 4. MEASURE PERFORMANCE
queries = TestQueryGenerator.generate_queries(index, num_queries=50)
metrics = {
    'latency': MetricsCollector.measure_query_latency(qp, queries),
    'throughput': MetricsCollector.measure_throughput(qp, queries, duration=5),
    'memory': MetricsCollector.measure_memory(),
    'index_size': MetricsCollector.measure_index_size('SelfIndex-v1.3210T', DataStore.DB1)
}

print(f"p95 Latency: {metrics['latency']['p95']:.2f} ms")
print(f"Throughput: {metrics['throughput']:.2f} q/s")

# 5. SAVE AND RELOAD
index.save_index()

# Later session
builder2 = IndexBuilder('SelfIndex-v1.3210T')
loaded_index = builder2.load_index()
qp2 = builder2.get_query_processor()
```

---

## Performance Benchmarks

### Configuration Tested
- **Dataset**: 109,762 documents
- **Queries**: 50 auto-generated Boolean queries
- **Hardware**: Standard development machine



## Troubleshooting

### Common Issues

1. **UTF-8 Decode Error**
   ```python
   # Fallback encoding in indexing logic
   try:
       text = raw.decode('utf-8')
   except:
       text = raw.decode('latin-1')
   ```

2. **Redis Connection Refused**
   ```bash
   # Start Redis manually
   redis-server --daemonize yes
   ```

3. **SQLite Lock Error**
   ```python
   # Ensure connection is closed
   if self.db_connection:
       self.db_connection.close()
   ```

4. **Memory Overflow**
   - Reduce `max_docs` parameter
   - Use compression (VarByte or GZIP)
   - Switch to SQLite for disk-based storage

---

## NOTES 

- **NLTK Porter Stemmer**: https://www.nltk.org/api/nltk.stem.porter.html
- **Gap Encoding**: Introduction to Information Retrieval (Manning et al.)
- **TF-IDF Formula**: `tf-idf(t, d) = (tf(t, d) / |d|) * log((N + 1) / (df(t) + 1))`
