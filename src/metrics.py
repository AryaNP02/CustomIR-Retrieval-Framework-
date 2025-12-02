"""
Metrics collection and reporting module
Collects performance metrics for index and query processing
"""

import time
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict

from .core import DataStore


class MetricsCollector:
    """Collect and analyze system metrics"""
    
    @staticmethod
    def measure_memory() -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def measure_index_size(index_name: str, datastore: DataStore) -> float:
        """
        Get index size on disk in MB
        Represents the actual memory footprint of the index
        """
        total_size = 0
        if datastore == DataStore.CUSTOM:
            path = Path(f"indices/{index_name}_custom.pkl")
        elif datastore == DataStore.DB1:
            path = Path(f"indices/{index_name}.db")
        elif datastore == DataStore.DB2:
            path = Path(f"indices/{index_name}_custom.pkl")
        else:
            return 0.0
        
        if path.exists():
            total_size = path.stat().st_size
        
        return total_size / 1024 / 1024

    
    @staticmethod
    def measure_query_latency(query_processor, 
                              queries: List[str], 
                              top_k: int = 10,
                              repetitions: int = 1) -> Dict:
        """
        Measure query latency statistics (averaged over multiple repetitions)
        Returns dict with averaged mean, p95, p99 latencies in milliseconds
        """
        all_results = []
        
        for _ in range(repetitions):
            latencies = []
            for query in queries:
                start_time = time.time()
                _ = query_processor.process_ranked_query(query, top_k)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            latencies.sort()
            stats = {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': min(latencies),
                'max': max(latencies),
                'std': np.std(latencies)
            }
            all_results.append(stats)
        
        # Average across repetitions
        averaged = {k: np.mean([res[k] for res in all_results]) for k in all_results[0]}
        return averaged

    @staticmethod
    def measure_throughput(query_processor, 
                        queries: List[str], 
                        duration: int = 10,
                        repetitions: int = 1) -> float:
        """
        Measure query throughput (queries per second).
        Runs queries continuously for the specified duration, repeated multiple times.
        Returns the average throughput across repetitions.
        """
        if not queries:
            return 0.0

        throughputs = []
        start_time = time.perf_counter()

        for _ in range(repetitions):
            query_count = 0
            for query in queries:
                _ = query_processor.process_ranked_query(query)
                query_count += 1
            throughputs.append(query_count)

        end_time = time.perf_counter()
        total_elapsed = end_time - start_time
        total_queries = sum(throughputs)

        throughput = total_queries / total_elapsed if total_elapsed > 0 else 0.0
        return throughput


class Reporter:
    """Generate reports and plots"""
    
    @staticmethod
    def print_metrics_report(version: str, metrics: Dict):
        """Print formatted metrics report"""
        print(f"\n{'='*70}")
        print(f"Metrics Report: {version}")
        print(f"{'='*70}")
        
        if 'latency' in metrics:
            print("\nLatency Statistics (ms):")
            print(f"  Mean:     {metrics['latency']['mean']:.2f}")
            print(f"  Median:   {metrics['latency']['median']:.2f}")
            print(f"  P95:      {metrics['latency']['p95']:.2f}")
            print(f"  P99:      {metrics['latency']['p99']:.2f}")
        
        if 'throughput' in metrics:
            print(f"\nThroughput: {metrics['throughput']:.2f} queries/second")
        
        if 'memory' in metrics:
            print(f"\nMemory Usage: {metrics['memory']:.2f} MB")
        
        if 'index_size' in metrics:
            print(f"Index Size on Disk: {metrics['index_size']:.2f} MB")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def compare_metrics(results: Dict[str, Dict]):
        """Compare metrics across different configurations"""
        print(f"\n{'='*100}")
        print(f"Comparison Report")
        print(f"{'='*100}")
        print(f"{'Version':<30} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<15} {'Memory (MB)':<15} {'Size (MB)':<12}")
        print(f"{'-'*100}")
        
        for version, metrics in results.items():
            p95 = metrics.get('latency', {}).get('p95', 0)
            p99 = metrics.get('latency', {}).get('p99', 0)
            throughput = metrics.get('throughput', 0)
            memory = metrics.get('memory', 0)
            size = metrics.get('index_size', 0)
            
            print(f"{version:<30} {p95:<12.2f} {p99:<12.2f} {throughput:<15.2f} {memory:<15.2f} {size:<12.2f}")
        
        print(f"{'='*100}\n")
