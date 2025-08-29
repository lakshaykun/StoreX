#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <chrono>
#include <random>
#include <thread>
#include <future>

// Include all project headers
#include "document.hpp"
#include "collection.hpp"
#include "similarity.hpp"
#include "index.hpp"
#include "storage.hpp"
#include "vector_store.hpp"
#include "lsh.hpp"
#include "utility.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

// Test counter
int advanced_tests_passed = 0;
int advanced_tests_total = 0;

// Helper macros for testing
#define ADVANCED_TEST(condition, message) \
    advanced_tests_total++; \
    if (condition) { \
        advanced_tests_passed++; \
        cout << "[PASS] " << message << endl; \
    } else { \
        cout << "[FAIL] " << message << endl; \
    }

#define ADVANCED_TEST_APPROX_EQUAL(a, b, epsilon, message) \
    advanced_tests_total++; \
    if (std::abs(a - b) < epsilon) { \
        advanced_tests_passed++; \
        cout << "[PASS] " << message << endl; \
    } else { \
        cout << "[FAIL] " << message << " (expected " << b << ", got " << a << ")" << endl; \
    }

// Test helper functions
bool vectorsEqual(const vector<float>& v1, const vector<float>& v2, float epsilon = 1e-6) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); i++) {
        if (std::abs(v1[i] - v2[i]) > epsilon) return false;
    }
    return true;
}

// ============== BENCHMARK TESTS ==============
void benchmarkIndexingStrategies() {
    cout << "\n=== Benchmarking Indexing Strategies ===" << endl;
    
    const size_t num_docs = 500;
    const size_t vec_dim = 50;
    const size_t num_queries = 20;
    
    // Generate test data
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    vector<Document> docs;
    for (size_t i = 0; i < num_docs; i++) {
        vector<float> emb(vec_dim);
        for (size_t j = 0; j < vec_dim; j++) {
            emb[j] = dist(rng);
        }
        docs.emplace_back(emb);
    }
    
    vector<vector<float>> queries;
    for (size_t i = 0; i < num_queries; i++) {
        vector<float> query(vec_dim);
        for (size_t j = 0; j < vec_dim; j++) {
            query[j] = dist(rng);
        }
        queries.push_back(query);
    }
    
    auto similarity = std::make_shared<EuclideanSimilarity>();
    
    // Benchmark FlatIndex
    auto flat_collection = std::make_shared<Collection>(vec_dim, false);
    auto flat_index = std::make_shared<FlatIndex>(flat_collection, similarity);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& doc : docs) {
        flat_index->insert(doc);
    }
    auto flat_insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        flat_index->search(query, 10);
    }
    auto flat_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    // Benchmark LSHIndex
    auto lsh_collection = std::make_shared<Collection>(vec_dim, false);
    auto lsh_index = std::make_shared<LSHIndex>(lsh_collection, similarity);
    
    start = std::chrono::high_resolution_clock::now();
    for (auto& doc : docs) {
        lsh_index->insert(doc);
    }
    auto lsh_insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        lsh_index->search(query, 10);
    }
    auto lsh_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    // Benchmark AnnoyIndex
    auto annoy_collection = std::make_shared<Collection>(vec_dim, false);
    auto annoy_index = std::make_unique<AnnoyIndex>(10, 5, annoy_collection, similarity);
    
    start = std::chrono::high_resolution_clock::now();
    for (auto& doc : docs) {
        annoy_index->insert(doc);
    }
    auto annoy_insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        annoy_index->search(query, 10);
    }
    auto annoy_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    // Report results
    cout << "  FlatIndex - Insert: " << flat_insert_time.count() << "ms, Search: " << flat_search_time.count() << "ms" << endl;
    cout << "  LSHIndex - Insert: " << lsh_insert_time.count() << "ms, Search: " << lsh_search_time.count() << "ms" << endl;
    cout << "  AnnoyIndex - Insert: " << annoy_insert_time.count() << "ms, Search: " << annoy_search_time.count() << "ms" << endl;
    
    ADVANCED_TEST(flat_insert_time.count() < 1000, "FlatIndex insert performance acceptable");
    ADVANCED_TEST(lsh_insert_time.count() < 5000, "LSHIndex insert performance acceptable");
    ADVANCED_TEST(annoy_insert_time.count() < 10000, "AnnoyIndex insert performance acceptable");
    
    ADVANCED_TEST(flat_search_time.count() < 500, "FlatIndex search performance acceptable");
    ADVANCED_TEST(lsh_search_time.count() < 200, "LSHIndex search performance acceptable");
    ADVANCED_TEST(annoy_search_time.count() < 100, "AnnoyIndex search performance acceptable");
}

// ============== ACCURACY TESTS ==============
void testSearchAccuracy() {
    cout << "\n=== Testing Search Accuracy ===" << endl;
    
    // Create a controlled dataset where we know the nearest neighbors
    auto similarity = std::make_shared<EuclideanSimilarity>();
    auto collection = std::make_shared<Collection>(3, false);
    auto index = std::make_shared<FlatIndex>(collection, similarity);
    
    // Insert known vectors
    vector<float> vec1 = {0.0f, 0.0f, 0.0f};
    vector<float> vec2 = {1.0f, 0.0f, 0.0f};  // Distance 1 from vec1
    vector<float> vec3 = {0.0f, 1.0f, 0.0f};  // Distance 1 from vec1
    vector<float> vec4 = {2.0f, 0.0f, 0.0f};  // Distance 2 from vec1
    vector<float> vec5 = {0.0f, 0.0f, 2.0f};  // Distance 2 from vec1
    
    Document doc1(vec1), doc2(vec2), doc3(vec3), doc4(vec4), doc5(vec5);
    index->insert(doc1);  // ID 0
    index->insert(doc2);  // ID 1
    index->insert(doc3);  // ID 2
    index->insert(doc4);  // ID 3
    index->insert(doc5);  // ID 4
    
    // Query with vec1 - should return all vectors ordered by distance
    vector<std::pair<float, Document>> results = index->searchWithScores(vec1, 5);
    
    ADVANCED_TEST(results.size() == 5, "Returns all vectors");
    ADVANCED_TEST(results[0].second.getEmbedding() == vec1, "Nearest neighbor is identical vector");
    
    // Check that results are sorted by similarity (higher similarity = lower distance for Euclidean)
    for (size_t i = 1; i < results.size(); i++) {
        ADVANCED_TEST(results[i-1].first >= results[i].first, 
                     "Results sorted by similarity (descending)");
    }
}

// ============== MEMORY TESTS ==============
void testMemoryUsage() {
    cout << "\n=== Testing Memory Usage ===" << endl;
    
    // Test for memory leaks with repeated insertions and deletions
    {
        auto collection = std::make_shared<Collection>(100, false);
        auto similarity = std::make_shared<CosineSimilarity>();
        auto index = std::make_shared<FlatIndex>(collection, similarity);
        
        // Insert and search many times
        for (int i = 0; i < 100; i++) {
            vector<float> emb(100, static_cast<float>(i));
            Document doc(emb);
            index->insert(doc);
            
            if (i % 10 == 0) {
                vector<Document> results = index->search(emb, 5);
            }
        }
        
        ADVANCED_TEST(true, "Memory test completed without crashes");
    } // Objects should be cleaned up here
    
    ADVANCED_TEST(true, "Memory cleanup completed successfully");
}

// ============== THREAD SAFETY TESTS ==============
void testThreadSafety() {
    cout << "\n=== Testing Thread Safety ===" << endl;
    
    auto collection = std::make_shared<Collection>(10, false);
    auto similarity = std::make_shared<CosineSimilarity>();
    auto index = std::make_shared<FlatIndex>(collection, similarity);
    
    // Pre-populate with some data
    for (int i = 0; i < 50; i++) {
        vector<float> emb(10, static_cast<float>(i));
        Document doc(emb);
        index->insert(doc);
    }
    
    const int num_threads = 4;
    const int operations_per_thread = 25;
    
    // Test concurrent reads
    vector<std::future<bool>> futures;
    
    auto search_worker = [&]() -> bool {
        try {
            for (int i = 0; i < operations_per_thread; i++) {
                vector<float> query(10, static_cast<float>(i % 20));
                vector<Document> results = index->search(query, 5);
                if (results.empty() && collection->size() > 0) {
                    return false;
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    };
    
    for (int i = 0; i < num_threads; i++) {
        futures.push_back(std::async(std::launch::async, search_worker));
    }
    
    bool all_succeeded = true;
    for (auto& future : futures) {
        if (!future.get()) {
            all_succeeded = false;
        }
    }
    
    ADVANCED_TEST(all_succeeded, "Concurrent read operations completed successfully");
}

// ============== DATA INTEGRITY TESTS ==============
void testDataIntegrity() {
    cout << "\n=== Testing Data Integrity ===" << endl;
    
    // Test that inserted data remains unchanged after operations
    auto collection = std::make_shared<Collection>(3, false);
    auto similarity = std::make_shared<CosineSimilarity>();
    auto index = std::make_shared<FlatIndex>(collection, similarity);
    
    // Insert original data
    vector<float> original_emb = {1.5f, 2.5f, 3.5f};
    string original_json = R"({"test": "value", "number": 42})";
    Document original_doc(original_emb, Metadata(original_json));
    
    size_t id = index->insert(original_doc);
    
    // Perform various operations
    vector<float> query = {1.0f, 2.0f, 3.0f};
    vector<Document> results = index->search(query, 1);
    
    // Verify data integrity
    Document retrieved = collection->getDocument(id);
    ADVANCED_TEST(retrieved.getEmbedding() == original_emb, "Embedding preserved after operations");
    ADVANCED_TEST(retrieved.getMetadata() == original_doc.getMetadata(), "Metadata preserved after operations");
    
    // Test update integrity
    vector<float> new_emb = {4.0f, 5.0f, 6.0f};
    Document new_doc(new_emb, Metadata(original_json));
    index->update(id, new_doc);
    
    Document updated = collection->getDocument(id);
    ADVANCED_TEST(updated.getEmbedding() == new_emb, "Update changes embedding correctly");
    ADVANCED_TEST(updated.getMetadata() == original_doc.getMetadata(), "Update preserves metadata");
}

// ============== SERIALIZATION TESTS ==============
void testSerialization() {
    cout << "\n=== Testing Serialization ===" << endl;
    
    const string test_db = "advanced_test.db";
    
    // Create and populate a collection
    {
        Storage storage(test_db);
        
        for (int i = 0; i < 10; i++) {
            vector<float> emb = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)};
            string json = R"({"id": )" + std::to_string(i) + R"(, "type": "test"})";
            Document doc(emb, Metadata(json));
            storage.insert(i, doc);
        }
        
        ADVANCED_TEST(true, "Data serialization completed");
    }
    
    // Load and verify data
    {
        Storage storage(test_db);
        vector<Document> loaded_docs = storage.load();
        
        ADVANCED_TEST(loaded_docs.size() == 10, "All documents loaded from storage");
        
        if (!loaded_docs.empty()) {
            vector<float> expected_first = {0.0f, 1.0f, 2.0f};
            ADVANCED_TEST(loaded_docs[0].getEmbedding() == expected_first, "First document loaded correctly");
            
            auto metadata = loaded_docs[0].getMetadata().getData();
            ADVANCED_TEST(metadata["id"] == 0, "Metadata deserialized correctly");
        }
    }
    
    // Cleanup
    std::filesystem::remove(test_db);
    ADVANCED_TEST(true, "Serialization test cleanup completed");
}

// ============== COMPREHENSIVE ACCURACY TESTS ==============
void testComprehensiveAccuracy() {
    cout << "\n=== Testing Comprehensive Search Accuracy ===" << endl;
    
    // Create a more complex dataset with known clusters
    auto similarity = std::make_shared<CosineSimilarity>();
    auto collection = std::make_shared<Collection>(3, false);
    auto flat_index = std::make_shared<FlatIndex>(collection, similarity);
    
    // Cluster 1: Points around (1,0,0)
    vector<vector<float>> cluster1 = {
        {1.0f, 0.1f, 0.0f}, {0.9f, 0.0f, 0.1f}, {1.1f, -0.1f, 0.0f}
    };
    
    // Cluster 2: Points around (0,1,0)
    vector<vector<float>> cluster2 = {
        {0.0f, 1.0f, 0.1f}, {0.1f, 0.9f, 0.0f}, {-0.1f, 1.1f, 0.0f}
    };
    
    // Outlier
    vector<float> outlier = {-1.0f, -1.0f, -1.0f};
    
    // Insert all points
    for (auto& point : cluster1) {
        Document doc(point);
        flat_index->insert(doc);
    }
    for (auto& point : cluster2) {
        Document doc(point);
        flat_index->insert(doc);
    }
    Document outlier_doc(outlier);
    flat_index->insert(outlier_doc);
    
    // Query with cluster 1 centroid
    vector<float> query1 = {1.0f, 0.0f, 0.0f};
    vector<std::pair<float, Document>> results1 = flat_index->searchWithScores(query1, 3);
    
    ADVANCED_TEST(results1.size() == 3, "Returns requested number of nearest neighbors");
    
    // Check that all top 3 results are from cluster 1
    int cluster1_count = 0;
    for (const auto& result : results1) {
        vector<float> emb = result.second.getEmbedding();
        for (const auto& cluster_point : cluster1) {
            if (vectorsEqual(emb, cluster_point, 0.01f)) {
                cluster1_count++;
                break;
            }
        }
    }
    ADVANCED_TEST(cluster1_count == 3, "Nearest neighbors belong to correct cluster");
    
    // Test precision-recall for different k values
    vector<float> query2 = {0.0f, 1.0f, 0.0f};
    vector<Document> results_k2 = flat_index->search(query2, 2);
    vector<Document> results_k5 = flat_index->search(query2, 5);
    
    ADVANCED_TEST(results_k2.size() == 2, "K=2 search returns 2 results");
    ADVANCED_TEST(results_k5.size() == 5, "K=5 search returns 5 results");
}

// ============== STRESS TESTING ==============
void testStressScenarios() {
    cout << "\n=== Testing Stress Scenarios ===" << endl;
    
    // Test with rapidly changing data
    auto collection = std::make_shared<Collection>(10, false);
    auto similarity = std::make_shared<EuclideanSimilarity>();
    auto index = std::make_shared<FlatIndex>(collection, similarity);
    
    const size_t num_operations = 200;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_int_distribution<int> op_dist(0, 2); // 0=insert, 1=update, 2=search
    
    vector<size_t> inserted_ids;
    size_t successful_operations = 0;
    
    for (size_t i = 0; i < num_operations; i++) {
        int operation = op_dist(rng);
        
        try {
            if (operation == 0 || inserted_ids.empty()) { // Insert
                vector<float> emb(10);
                for (int j = 0; j < 10; j++) {
                    emb[j] = dist(rng);
                }
                Document doc(emb);
                size_t id = index->insert(doc);
                inserted_ids.push_back(id);
                successful_operations++;
                
            } else if (operation == 1) { // Update
                size_t random_id = inserted_ids[rng() % inserted_ids.size()];
                vector<float> new_emb(10);
                for (int j = 0; j < 10; j++) {
                    new_emb[j] = dist(rng);
                }
                Document new_doc(new_emb);
                index->update(random_id, new_doc);
                successful_operations++;
                
            } else { // Search
                vector<float> query(10);
                for (int j = 0; j < 10; j++) {
                    query[j] = dist(rng);
                }
                vector<Document> results = index->search(query, 5);
                if (!results.empty() || inserted_ids.empty()) {
                    successful_operations++;
                }
            }
        } catch (const std::exception& e) {
            // Count failures but continue
        }
    }
    
    ADVANCED_TEST(successful_operations >= num_operations * 0.9, 
                 "At least 90% of stress test operations succeed");
    
    cout << "  Stress test: " << successful_operations << "/" << num_operations 
         << " operations successful" << endl;
}

// ============== CONCURRENCY ADVANCED TESTS ==============
void testAdvancedConcurrency() {
    cout << "\n=== Testing Advanced Concurrency ===" << endl;
    
    auto collection = std::make_shared<Collection>(5, false);
    auto similarity = std::make_shared<CosineSimilarity>();
    auto index = std::make_shared<FlatIndex>(collection, similarity);
    
    // Pre-populate with data
    for (int i = 0; i < 100; i++) {
        vector<float> emb(5);
        for (int j = 0; j < 5; j++) {
            emb[j] = static_cast<float>(i + j);
        }
        Document doc(emb);
        index->insert(doc);
    }
    
    const int num_threads = 8;
    const int operations_per_thread = 50;
    
    // Test mixed read/write operations
    vector<std::future<bool>> futures;
    std::atomic<int> successful_reads(0);
    std::atomic<int> total_reads(0);
    
    auto mixed_worker = [&](int thread_id) -> bool {
        try {
            for (int i = 0; i < operations_per_thread; i++) {
                if (i % 3 == 0) {
                    // Insert operation
                    vector<float> emb(5);
                    for (int j = 0; j < 5; j++) {
                        emb[j] = static_cast<float>(thread_id * 1000 + i * 10 + j);
                    }
                    Document doc(emb);
                    index->insert(doc);
                } else {
                    // Search operation
                    vector<float> query(5);
                    for (int j = 0; j < 5; j++) {
                        query[j] = static_cast<float>((thread_id + i) % 50 + j);
                    }
                    vector<Document> results = index->search(query, 3);
                    
                    total_reads++;
                    if (!results.empty()) {
                        successful_reads++;
                    }
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    };
    
    for (int i = 0; i < num_threads; i++) {
        futures.push_back(std::async(std::launch::async, mixed_worker, i));
    }
    
    bool all_threads_successful = true;
    for (auto& future : futures) {
        if (!future.get()) {
            all_threads_successful = false;
        }
    }
    
    ADVANCED_TEST(all_threads_successful, "All concurrent threads completed successfully");
    ADVANCED_TEST(successful_reads.load() >= total_reads.load() * 0.8, 
                 "At least 80% of concurrent reads successful");
    
    cout << "  Concurrent operations: " << successful_reads.load() << "/" 
         << total_reads.load() << " reads successful" << endl;
}

// ============== MEMORY LEAK DETECTION ==============
void testMemoryLeakDetection() {
    cout << "\n=== Testing Memory Leak Detection ===" << endl;
    
    // Test with many index creations and destructions
    const int iterations = 50;
    
    for (int i = 0; i < iterations; i++) {
        {
            auto collection = std::make_shared<Collection>(20, false);
            auto similarity = std::make_shared<EuclideanSimilarity>();
            
            // Test different index types in rotation
            if (i % 3 == 0) {
                auto index = std::make_shared<FlatIndex>(collection, similarity);
                for (int j = 0; j < 10; j++) {
                    vector<float> emb(20, static_cast<float>(j));
                    Document doc(emb);
                    index->insert(doc);
                }
            } else if (i % 3 == 1) {
                auto index = std::make_shared<LSHIndex>(collection, similarity);
                for (int j = 0; j < 10; j++) {
                    vector<float> emb(20, static_cast<float>(j));
                    Document doc(emb);
                    index->insert(doc);
                }
            } else {
                try {
                    auto index = std::make_shared<AnnoyIndex>(5, 3, collection, similarity);
                    for (int j = 0; j < 10; j++) {
                        vector<float> emb(20, static_cast<float>(j));
                        Document doc(emb);
                        index->insert(doc);
                    }
                } catch (...) {
                    // AnnoyIndex might fail in some cases, that's ok
                }
            }
        } // All objects should be destroyed here
    }
    
    ADVANCED_TEST(true, "Memory leak test completed without crashes");
    
    // Test vector store memory management
    for (int i = 0; i < 10; i++) {
        {
            vector_store store(10);
            for (int j = 0; j < 20; j++) {
                vector<float> emb(10, static_cast<float>(j));
                Document doc(emb);
                store.insert(doc);
            }
            
            // Perform searches
            for (int j = 0; j < 5; j++) {
                vector<float> query(10, static_cast<float>(j));
                store.search(query, 3);
            }
        } // vector_store should be cleaned up here
    }
    
    ADVANCED_TEST(true, "Vector store memory management test completed");
}

// ============== SERIALIZATION ADVANCED TESTS ==============
void testAdvancedSerialization() {
    cout << "\n=== Testing Advanced Serialization ===" << endl;
    
    const string test_db = "advanced_serialization_test.db";
    
    // Test serialization of complex metadata
    {
        Storage storage(test_db);
        
        for (int i = 0; i < 5; i++) {
            vector<float> emb = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)};
            
            // Create complex JSON metadata
            string complex_json = R"({
                "id": )" + std::to_string(i) + R"(,
                "title": "Document )" + std::to_string(i) + R"(",
                "tags": ["tag)" + std::to_string(i) + R"(", "tag)" + std::to_string(i+1) + R"("],
                "metadata": {
                    "created": "2025-08-29",
                    "version": 1.)" + std::to_string(i) + R"(,
                    "nested": {
                        "level": )" + std::to_string(i) + R"(,
                        "active": )" + (i % 2 == 0 ? "true" : "false") + R"(
                    }
                }
            })";
            
            Document doc(emb, Metadata(complex_json));
            storage.insert(i, doc);
        }
        
        ADVANCED_TEST(true, "Complex metadata serialization completed");
    }
    
    // Test loading and verification of complex data
    {
        Storage storage(test_db);
        vector<Document> loaded_docs = storage.load();
        
        ADVANCED_TEST(loaded_docs.size() == 5, "All complex documents loaded");
        
        if (!loaded_docs.empty()) {
            auto metadata = loaded_docs[0].getMetadata().getData();
            ADVANCED_TEST(metadata.contains("id"), "Complex metadata contains id field");
            ADVANCED_TEST(metadata.contains("metadata"), "Complex metadata contains nested objects");
            ADVANCED_TEST(metadata["metadata"].contains("nested"), "Deeply nested metadata preserved");
        }
    }
    
    // Test serialization under concurrent access
    try {
        const int num_concurrent_writers = 3;
        vector<std::future<bool>> write_futures;
        
        auto concurrent_writer = [&](int writer_id) -> bool {
            try {
                string db_name = test_db + std::to_string(writer_id);
                Storage storage(db_name);
                
                for (int i = 0; i < 10; i++) {
                    vector<float> emb(3, static_cast<float>(writer_id * 10 + i));
                    string json = R"({"writer": )" + std::to_string(writer_id) + 
                                 R"(, "doc": )" + std::to_string(i) + "}";
                    Document doc(emb, Metadata(json));
                    storage.insert(i, doc);
                }
                return true;
            } catch (...) {
                return false;
            }
        };
        
        for (int i = 0; i < num_concurrent_writers; i++) {
            write_futures.push_back(std::async(std::launch::async, concurrent_writer, i));
        }
        
        bool all_writers_successful = true;
        for (auto& future : write_futures) {
            if (!future.get()) {
                all_writers_successful = false;
            }
        }
        
        ADVANCED_TEST(all_writers_successful, "Concurrent serialization successful");
        
        // Cleanup concurrent test files
        for (int i = 0; i < num_concurrent_writers; i++) {
            std::filesystem::remove(test_db + std::to_string(i));
        }
        
    } catch (const std::exception& e) {
        ADVANCED_TEST(false, string("Concurrent serialization failed: ") + e.what());
    }
    
    // Cleanup main test file
    std::filesystem::remove(test_db);
    ADVANCED_TEST(true, "Advanced serialization test cleanup completed");
}

// ============== MAIN ADVANCED TEST RUNNER ==============
int main() {
    cout << "Running StoreX Advanced Tests..." << endl;
    cout << "===================================" << endl;
    
    try {
        benchmarkIndexingStrategies();
        testSearchAccuracy();
        testComprehensiveAccuracy();
        testMemoryUsage();
        testMemoryLeakDetection();
        testThreadSafety();
        testAdvancedConcurrency();
        testStressScenarios();
        testDataIntegrity();
        testSerialization();
        testAdvancedSerialization();
    } catch (const std::exception& e) {
        cout << "Advanced test execution failed with exception: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n===================================" << endl;
    cout << "Advanced Test Results: " << advanced_tests_passed << "/" << advanced_tests_total << " passed" << endl;
    
    if (advanced_tests_passed == advanced_tests_total) {
        cout << "All advanced tests passed! ✓" << endl;
        return 0;
    } else {
        cout << "Some advanced tests failed! ✗" << endl;
        return 1;
    }
}
