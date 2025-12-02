## Quick Start Guide

### Project Structure

```
selfindex/
├── src/                               # Implementation (do not modify for inference)
│   ├── __init__.py                   # Package exports
│   ├── core.py                       # Enums & compression
│   ├── preprocessor.py               # Text preprocessing
│   ├── index.py                      # InvertedIndex class
│   ├── query_processor.py            # Boolean & ranked query
│   ├── metrics.py                    # Performance metrics
│   └── index_builder.py              # Index builder
│
├── config/                            # Configuration files (EDIT THESE)
│   ├── index_config.yaml             # Main parameters
│   └── version_mapping.yaml          # Version reference
│
├── inference.ipynb                    # ✓ Run this notebook for inference/evaluation
├── requirements.txt                   # Python dependencies
├── setup.py                           # Setup script
├── README.md                          # Full documentation
└── QUICKSTART.md                      # This file
```

### Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Or run setup script**:
   ```bash
   python setup.py
   ```

### Configuration

Edit `config/index_config.yaml` to customize:

```yaml
data:
  directory: "free-news-datasets/News_Datasets"  # Your data path
  max_docs: 1e6                                   # Documents to index

index:
  type: "BOOLEAN"                # BOOLEAN, WORDCOUNT, TFIDF
  storage: "CUSTOM"               # CUSTOM, SQLITE, REDIS
  compression: "NONE"             # NONE, VARBYTE_ENCODING, GZIP_COMPRESSION
  optimization: "NULL"            # NULL, SKIPPING, THRESHOLDING, EARLY_STOPPING
  query_strategy: "TERM_AT_A_TIME" # TERM_AT_A_TIME, DOCUMENT_AT_A_TIME

query:
  default_top_k: 10               # Results to return
```

### Usage

1. **Open the notebook**:
   ```bash
   jupyter notebook inference.ipynb
   ```

2. **Follow the notebook sections**:
   - **Section 1**: Understand project structure
   - **Section 2**: Import modules (automatic)
   - **Section 3**: Load configuration from YAML
   - **Section 4**: Initialize index & query processor
   - **Section 5**: Run queries and inference
   - **Section 6**: Generate performance metrics & visualizations

### Example Workflow

```python
# All done in the notebook!

# 1. Configuration is loaded automatically
# 2. Build or load index
index = builder.load_index()  # or builder.build_index(DATA_DIR, max_docs=1000)

# 3. Run queries
results = qp.process_ranked_query("technology innovation", top_k=10)
for doc in results:
    print(doc['title'], doc['score'])

# 4. Measure performance
metrics = {
    'latency': MetricsCollector.measure_query_latency(qp, queries),
    'throughput': MetricsCollector.measure_throughput(qp, queries),
    'memory': MetricsCollector.measure_memory(),
    'index_size': MetricsCollector.measure_index_size(version, datastore)
}

# 5. Generate report
Reporter.print_metrics_report(version, metrics)
```

### Version String Format

`SelfIndex-v1.xyziqQ`

- **x**: IndexInfo (1=Boolean, 2=WordCount, 3=TF-IDF)
- **y**: DataStore (1=Custom, 2=SQLite, 3=Redis)
- **z**: Compression (1=None, 2=VarByte, 3=GZIP)
- **i**: Optimization (0=Null, sp=Skipping, th=Thresholding, es=EarlyStopping)
- **q**: QueryProc (T=Term-at-a-time, D=Document-at-a-time)

Example: `SelfIndex-v1.321spT` = TF-IDF + SQLite + VarByte + Skipping + Term-at-a-time

### Common Tasks

#### Build a new index
```python
# Uncomment in Section 4 of notebook
if DATA_DIR.exists():
    index = builder.build_index(DATA_DIR, max_docs=MAX_DOCS)
```

#### Load existing index
```python
# Already in notebook Section 4
index = builder.load_index()
```

#### Run boolean queries
```python
results = qp.process_boolean_query('("machine" AND "learning") OR "ai"')
print(len(results))  # Number of matching documents
```

#### Benchmark multiple configurations
```python
# Uncomment last section of notebook for multi-config benchmarking
```

### Troubleshooting

**Index not found**: Make sure to build index first or check `indices/` directory

**Import errors**: Run `python setup.py` to download NLTK data

**Config not loaded**: Verify `config/index_config.yaml` exists and is valid YAML

**Data directory not found**: Update `DATA_DIR` in `config/index_config.yaml`

### File Organization Best Practice

- **Don't edit** `src/` files for inference experiments
- **Do edit** `config/index_config.yaml` to change parameters
- **Keep** inference code in `inference.ipynb`
- **Save** results to `plot/` directory

### Key Classes (see README.md for details)

- `IndexBuilder`: Build/load indices
- `QueryProcessor`: Process queries
- `MetricsCollector`: Measure performance
- `TestQueryGenerator`: Generate test queries
- `InvertedIndex`: Core index structure

See `README.md` for complete API documentation.
