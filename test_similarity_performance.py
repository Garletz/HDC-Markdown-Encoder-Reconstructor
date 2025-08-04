#!/usr/bin/env python3
"""Performance test for similarity search functionality."""

import time
import numpy as np
from hdc_markdown_transformer.storage.vector_database import InMemoryVectorDatabase
from hdc_markdown_transformer.storage.similarity_search import (
    SimilaritySearchEngine, 
    BatchSimilaritySearchEngine,
    SearchConfig
)

def test_similarity_search_performance():
    """Test performance of similarity search with various dataset sizes."""
    
    print("Testing Similarity Search Performance")
    print("=" * 50)
    
    # Test different dataset sizes
    dataset_sizes = [100, 500, 1000, 5000]
    vector_dimension = 100
    
    for size in dataset_sizes:
        print(f"\nTesting with {size} vectors (dimension {vector_dimension})")
        
        # Create test dataset
        db = InMemoryVectorDatabase()
        test_vectors = {}
        
        for i in range(size):
            vector = np.random.randn(vector_dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize
            test_vectors[f"token_{i}"] = vector
        
        db.store_vectors(test_vectors)
        search_engine = SimilaritySearchEngine(db)
        
        # Test single search performance
        query_vector = np.random.randn(vector_dimension).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        start_time = time.time()
        results = search_engine.search(query_vector, SearchConfig(k=10))
        single_search_time = time.time() - start_time
        
        print(f"  Single search (k=10): {single_search_time:.4f}s")
        print(f"  Results returned: {len(results)}")
        
        # Test batch search performance
        query_vectors = [np.random.randn(vector_dimension).astype(np.float32) for _ in range(10)]
        query_vectors = [v / np.linalg.norm(v) for v in query_vectors]
        
        start_time = time.time()
        batch_results = search_engine.batch_search(query_vectors, SearchConfig(k=5, batch_size=5))
        batch_search_time = time.time() - start_time
        
        print(f"  Batch search (10 queries, k=5): {batch_search_time:.4f}s")
        print(f"  Average per query: {batch_search_time/10:.4f}s")
        
        # Test with BatchSimilaritySearchEngine
        batch_engine = BatchSimilaritySearchEngine(db)
        
        start_time = time.time()
        parallel_results = batch_engine.parallel_batch_search(query_vectors, k=5, batch_size=5)
        parallel_search_time = time.time() - start_time
        
        print(f"  Parallel batch search: {parallel_search_time:.4f}s")
        print(f"  Speedup vs regular batch: {batch_search_time/parallel_search_time:.2f}x")

def test_cosine_similarity_accuracy():
    """Test accuracy of cosine similarity calculations."""
    
    print("\nTesting Cosine Similarity Accuracy")
    print("=" * 50)
    
    db = InMemoryVectorDatabase()
    
    # Create vectors with known cosine similarities
    reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_vectors = {
        "identical": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "similar_90": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "similar_80": np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32),
        "orthogonal": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "opposite": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }
    
    # Normalize vectors
    for key, vector in test_vectors.items():
        test_vectors[key] = vector / np.linalg.norm(vector)
    
    db.store_vectors(test_vectors)
    search_engine = SimilaritySearchEngine(db)
    
    # Search with reference vector
    results = search_engine.search(reference, SearchConfig(k=5))
    
    print("Expected vs Actual Cosine Similarities:")
    expected_similarities = {
        "identical": 1.0,
        "similar_90": np.dot(reference, test_vectors["similar_90"]),
        "similar_80": np.dot(reference, test_vectors["similar_80"]),
        "orthogonal": 0.0,
        "opposite": -1.0
    }
    
    for result in results:
        token = result.token
        actual_sim = result.similarity_score
        expected_sim = expected_similarities[token]
        error = abs(actual_sim - expected_sim)
        
        print(f"  {token:12}: Expected {expected_sim:.4f}, Got {actual_sim:.4f}, Error {error:.6f}")
        
        # Check accuracy (allow small floating point errors)
        assert error < 1e-5, f"Similarity calculation error too large for {token}"

def test_ranking_quality():
    """Test quality of result ranking."""
    
    print("\nTesting Ranking Quality")
    print("=" * 50)
    
    db = InMemoryVectorDatabase()
    
    # Create vectors with decreasing similarity to reference
    reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_vectors = {}
    
    for i in range(10):
        similarity = 1.0 - (i * 0.1)  # Decreasing from 1.0 to 0.1
        vector = np.array([similarity, np.sqrt(1 - similarity**2), 0.0, 0.0], dtype=np.float32)
        test_vectors[f"token_{i}"] = vector
    
    db.store_vectors(test_vectors)
    search_engine = SimilaritySearchEngine(db)
    
    # Search and verify ranking
    results = search_engine.search(reference, SearchConfig(k=10))
    
    print("Ranking verification:")
    for i, result in enumerate(results):
        expected_token = f"token_{i}"
        actual_token = result.token
        similarity = result.similarity_score
        
        print(f"  Rank {i+1}: {actual_token} (similarity: {similarity:.4f})")
        
        # Verify correct ranking order
        assert actual_token == expected_token, f"Ranking error at position {i}: expected {expected_token}, got {actual_token}"
    
    # Verify similarities are in descending order
    similarities = [r.similarity_score for r in results]
    assert similarities == sorted(similarities, reverse=True), "Results not properly sorted by similarity"
    
    print("  ✓ All results correctly ranked by similarity")

if __name__ == "__main__":
    test_similarity_search_performance()
    test_cosine_similarity_accuracy()
    test_ranking_quality()
    print("\n" + "=" * 50)
    print("All performance and accuracy tests passed!")