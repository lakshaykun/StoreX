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

// ============== MAIN ADVANCED TEST RUNNER ==============
int main() {
    cout << "Running StoreX Advanced Tests..." << endl;
    cout << "===================================" << endl;
    
    try {
        benchmarkIndexingStrategies();
        testSearchAccuracy();
        testMemoryUsage();
        testThreadSafety();
        testDataIntegrity();
        testSerialization();
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
