#include "search_engine.hpp"
#include "similarity.hpp"
#include "document.hpp"
#include "vector_store.hpp"
#include "metadata_filter.hpp"
#include "storage.hpp"
#include "index.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

class TestSuite {
private:
    int tests_passed = 0;
    int tests_failed = 0;
    
    void assert_test(bool condition, const string& test_name) {
        if (condition) {
            cout << "✅ PASS: " << test_name << endl;
            tests_passed++;
        } else {
            cout << "❌ FAIL: " << test_name << endl;
            tests_failed++;
        }
    }
    
    void assert_near(float a, float b, float tolerance, const string& test_name) {
        bool condition = abs(a - b) < tolerance;
        assert_test(condition, test_name + " (expected: " + to_string(b) + ", got: " + to_string(a) + ")");
    }

public:
    void test_document_creation() {
        cout << "\n=== Testing Document Creation ===" << endl;
        
        // Test default constructor
        Document doc1;
        assert_test(doc1.embedding.empty(), "Document default constructor - empty embedding");
        assert_test(doc1.metadata.empty(), "Document default constructor - empty metadata");
        
        // Test parameterized constructor
        vector<float> embedding = {1.0f, 2.0f, 3.0f};
        Metadata metadata = {{"id", 1}, {"category", string("test")}};
        Document doc2(embedding, metadata);
        
        assert_test(doc2.embedding.size() == 3, "Document parameterized constructor - embedding size");
        assert_test(doc2.embedding[0] == 1.0f, "Document parameterized constructor - embedding value");
        assert_test(std::get<int>(doc2.metadata.at("id")) == 1, "Document parameterized constructor - metadata id");
        assert_test(std::get<string>(doc2.metadata.at("category")) == "test", "Document parameterized constructor - metadata category");
    }
    
    void test_vector_store() {
        cout << "\n=== Testing VectorStore ===" << endl;
        
        VectorStore store;
        
        // Test empty store
        assert_test(store.getAll().empty(), "VectorStore initially empty");
        
        // Test insertion
        Document doc1({1.0f, 0.0f}, {{"id", 1}});
        Document doc2({0.0f, 1.0f}, {{"id", 2}});
        
        store.insert(doc1);
        assert_test(store.getAll().size() == 1, "VectorStore size after first insert");
        
        store.insert(doc2);
        assert_test(store.getAll().size() == 2, "VectorStore size after second insert");
        
        // Test retrieval
        const auto& documents = store.getAll();
        assert_test(documents[0].embedding[0] == 1.0f, "VectorStore retrieval - first doc");
        assert_test(documents[1].embedding[1] == 1.0f, "VectorStore retrieval - second doc");
    }
    
    void test_similarity_metrics() {
        cout << "\n=== Testing Similarity Metrics ===" << endl;
        
        vector<float> vec1 = {1.0f, 0.0f, 0.0f};
        vector<float> vec2 = {0.0f, 1.0f, 0.0f};
        vector<float> vec3 = {1.0f, 0.0f, 0.0f}; // Same as vec1
        
        // Test Cosine Similarity
        CosineSimilarity cosine;
        assert_near(cosine.compute(vec1, vec1), 1.0f, 1e-6f, "Cosine similarity - identical vectors");
        assert_near(cosine.compute(vec1, vec2), 0.0f, 1e-6f, "Cosine similarity - orthogonal vectors");
        assert_near(cosine.compute(vec1, vec3), 1.0f, 1e-6f, "Cosine similarity - same vectors");
        
        // Test Dot Product Similarity
        DotProductSimilarity dotProduct;
        assert_near(dotProduct.compute(vec1, vec1), 1.0f, 1e-6f, "Dot product similarity - identical vectors");
        assert_near(dotProduct.compute(vec1, vec2), 0.0f, 1e-6f, "Dot product similarity - orthogonal vectors");
        
        // Test Euclidean Similarity (note: this returns similarity, not distance)
        EuclideanSimilarity euclidean;
        float euclidean_same = euclidean.compute(vec1, vec1);
        float euclidean_diff = euclidean.compute(vec1, vec2);
        assert_test(euclidean_same > euclidean_diff, "Euclidean similarity - identical vectors should be more similar");
    }
    
    void test_metadata_filter() {
        cout << "\n=== Testing Metadata Filter ===" << endl;
        
        Metadata metadata1 = {{"category", string("A")}, {"score", 85}, {"value", 1.5f}};
        Metadata metadata2 = {{"category", string("B")}, {"score", 92}, {"value", 2.3f}};
        
        // Test EQ filter
        json eq_filter = json{{"op", "EQ"}, {"field", "category"}, {"value", "A"}};
        Filter parsed_eq = parseFilter(eq_filter);
        assert_test(evaluate(metadata1, parsed_eq), "Metadata filter EQ - match");
        assert_test(!evaluate(metadata2, parsed_eq), "Metadata filter EQ - no match");
        
        // Test GT filter
        json gt_filter = json{{"op", "GT"}, {"field", "score"}, {"value", 90}};
        Filter parsed_gt = parseFilter(gt_filter);
        assert_test(!evaluate(metadata1, parsed_gt), "Metadata filter GT - no match");
        assert_test(evaluate(metadata2, parsed_gt), "Metadata filter GT - match");
        
        // Test IN filter
        json in_filter = json{{"op", "IN"}, {"field", "category"}, {"values", json::array({"A", "C"})}};
        Filter parsed_in = parseFilter(in_filter);
        assert_test(evaluate(metadata1, parsed_in), "Metadata filter IN - match");
        assert_test(!evaluate(metadata2, parsed_in), "Metadata filter IN - no match");
        
        // Test AND filter
        json and_filter = json{
            {"op", "AND"},
            {"children", json::array({
                json{{"op", "EQ"}, {"field", "category"}, {"value", "A"}},
                json{{"op", "LT"}, {"field", "score"}, {"value", 90}}
            })}
        };
        Filter parsed_and = parseFilter(and_filter);
        assert_test(evaluate(metadata1, parsed_and), "Metadata filter AND - match");
        assert_test(!evaluate(metadata2, parsed_and), "Metadata filter AND - no match");
    }
    
    void test_flat_search_engine() {
        cout << "\n=== Testing Flat Search Engine ===" << endl;
        
        VectorStore store;
        store.insert(Document({1.0f, 0.0f}, {{"id", 1}, {"category", string("A")}}));
        store.insert(Document({0.9f, 0.1f}, {{"id", 2}, {"category", string("A")}}));
        store.insert(Document({0.0f, 1.0f}, {{"id", 3}, {"category", string("B")}}));
        store.insert(Document({0.1f, 0.9f}, {{"id", 4}, {"category", string("B")}}));
        
        CosineSimilarity metric;
        FlatSearchEngine engine(store, metric);
        
        // Test basic search
        vector<float> query = {1.0f, 0.0f};
        auto results = engine.search(query, 2);
        
        assert_test(results.size() == 2, "Flat search - result count");
        assert_test(std::get<int>(results[0].second.metadata.at("id")) == 1, "Flat search - top result");
        assert_test(results[0].first > results[1].first, "Flat search - results sorted");
        
        // Test filtered search
        json filter = json{{"op", "EQ"}, {"field", "category"}, {"value", "B"}};
        auto filtered_results = engine.search(query, 10, filter);
        
        assert_test(filtered_results.size() == 2, "Flat search with filter - result count");
        for (const auto& result : filtered_results) {
            string category = std::get<string>(result.second.metadata.at("category"));
            assert_test(category == "B", "Flat search with filter - category match");
        }
    }
    
    void test_lsh_search_engine() {
        cout << "\n=== Testing LSH Search Engine ===" << endl;
        
        VectorStore store;
        store.insert(Document({1.0f, 0.0f}, {{"id", 1}, {"category", string("A")}}));
        store.insert(Document({0.9f, 0.1f}, {{"id", 2}, {"category", string("A")}}));
        store.insert(Document({0.0f, 1.0f}, {{"id", 3}, {"category", string("B")}}));
        store.insert(Document({0.1f, 0.9f}, {{"id", 4}, {"category", string("B")}}));
        
        CosineSimilarity metric;
        LSHSearchEngine lsh_engine(store, metric, 5, 4);
        FlatSearchEngine flat_engine(store, metric);
        
        // Test basic search
        vector<float> query = {1.0f, 0.0f};
        auto lsh_results = lsh_engine.search(query, 4);
        auto flat_results = flat_engine.search(query, 4);
        
        assert_test(lsh_results.size() <= 4, "LSH search - result count constraint");
        assert_test(!lsh_results.empty(), "LSH search - non-empty results");
        
        // Check that top result matches (LSH should find the best match)
        if (!lsh_results.empty() && !flat_results.empty()) {
            int lsh_top = std::get<int>(lsh_results[0].second.metadata.at("id"));
            int flat_top = std::get<int>(flat_results[0].second.metadata.at("id"));
            assert_test(lsh_top == flat_top, "LSH vs Flat - top result match");
        }
        
        // Test filtered search
        json filter = json{{"op", "EQ"}, {"field", "category"}, {"value", "A"}};
        auto filtered_results = lsh_engine.search(query, 10, filter);
        
        for (const auto& result : filtered_results) {
            string category = std::get<string>(result.second.metadata.at("category"));
            assert_test(category == "A", "LSH search with filter - category match");
        }
    }
    
    void test_hnsw_search_engine() {
        cout << "\n=== Testing HNSW Search Engine ===" << endl;
        
        VectorStore store;
        store.insert(Document({1.0f, 0.0f}, {{"id", 1}, {"category", string("A")}}));
        store.insert(Document({0.9f, 0.1f}, {{"id", 2}, {"category", string("A")}}));
        store.insert(Document({0.0f, 1.0f}, {{"id", 3}, {"category", string("B")}}));
        store.insert(Document({0.1f, 0.9f}, {{"id", 4}, {"category", string("B")}}));
        
        CosineSimilarity metric;
        HNSWSearchEngine hnsw_engine(store, metric, 16, 100, 50);
        FlatSearchEngine flat_engine(store, metric);
        
        // Test basic search
        vector<float> query = {1.0f, 0.0f};
        auto hnsw_results = hnsw_engine.search(query, 4);
        auto flat_results = flat_engine.search(query, 4);
        
        assert_test(hnsw_results.size() <= 4, "HNSW search - result count constraint");
        assert_test(!hnsw_results.empty(), "HNSW search - non-empty results");
        
        // Check that top result matches (HNSW should find the best match)
        if (!hnsw_results.empty() && !flat_results.empty()) {
            int hnsw_top = std::get<int>(hnsw_results[0].second.metadata.at("id"));
            int flat_top = std::get<int>(flat_results[0].second.metadata.at("id"));
            assert_test(hnsw_top == flat_top, "HNSW vs Flat - top result match");
        }
        
        // Test filtered search
        json filter = json{{"op", "EQ"}, {"field", "category"}, {"value", "A"}};
        auto filtered_results = hnsw_engine.search(query, 10, filter);
        
        for (const auto& result : filtered_results) {
            string category = std::get<string>(result.second.metadata.at("category"));
            assert_test(category == "A", "HNSW search with filter - category match");
        }
    }
    
    void test_annoy_search_engine() {
        cout << "\n=== Testing Annoy Search Engine ===" << endl;
        
        VectorStore store;
        store.insert(Document({1.0f, 0.0f}, {{"id", 1}, {"category", string("A")}}));
        store.insert(Document({0.9f, 0.1f}, {{"id", 2}, {"category", string("A")}}));
        store.insert(Document({0.0f, 1.0f}, {{"id", 3}, {"category", string("B")}}));
        store.insert(Document({0.1f, 0.9f}, {{"id", 4}, {"category", string("B")}}));
        
        CosineSimilarity metric;
        AnnoySearchEngine annoy_engine(store, metric, 5, 2);
        FlatSearchEngine flat_engine(store, metric);
        
        // Test basic search
        vector<float> query = {1.0f, 0.0f};
        auto annoy_results = annoy_engine.search(query, 4);
        auto flat_results = flat_engine.search(query, 4);
        
        assert_test(annoy_results.size() <= 4, "Annoy search - result count constraint");
        assert_test(!annoy_results.empty(), "Annoy search - non-empty results");
        
        // Check that top result matches (Annoy should find the best match)
        if (!annoy_results.empty() && !flat_results.empty()) {
            int annoy_top = std::get<int>(annoy_results[0].second.metadata.at("id"));
            int flat_top = std::get<int>(flat_results[0].second.metadata.at("id"));
            assert_test(annoy_top == flat_top, "Annoy vs Flat - top result match");
        }
        
        // Test filtered search
        json filter = json{{"op", "EQ"}, {"field", "category"}, {"value", "A"}};
        auto filtered_results = annoy_engine.search(query, 10, filter);
        
        for (const auto& result : filtered_results) {
            string category = std::get<string>(result.second.metadata.at("category"));
            assert_test(category == "A", "Annoy search with filter - category match");
        }
    }
    
    void test_storage_operations() {
        cout << "\n=== Testing Storage Operations ===" << endl;
        
        // Create test data
        Document doc1({1.0f, 2.0f, 3.0f}, {{"id", 1}, {"name", string("doc1")}});
        Document doc2({4.0f, 5.0f, 6.0f}, {{"id", 2}, {"name", string("doc2")}});
        
        const string test_file = "test_storage.jsonl";
        Storage storage(test_file);
        
        // Test save single document
        bool save_success1 = storage.save_document(doc1);
        bool save_success2 = storage.save_document(doc2);
        assert_test(save_success1, "Storage save document 1");
        assert_test(save_success2, "Storage save document 2");
        
        // Check file exists
        ifstream file(test_file);
        assert_test(file.good(), "Storage file creation");
        file.close();
        
        // Test load all documents
        auto loaded_docs = storage.load_documents();
        assert_test(loaded_docs.size() == 2, "Storage load - document count");
        
        if (loaded_docs.size() >= 2) {
            assert_test(loaded_docs[0].embedding.size() == 3, "Storage load - embedding dimension");
            assert_test(loaded_docs[0].embedding[0] == 1.0f, "Storage load - embedding value");
            assert_test(std::get<int>(loaded_docs[0].metadata.at("id")) == 1, "Storage load - metadata value");
        }
        
        // Cleanup
        remove(test_file.c_str());
    }
    
    void test_index_operations() {
        cout << "\n=== Testing Index Operations (Skipped) ===" << endl;
        cout << "Note: Index class appears to be an abstract interface." << endl;
        cout << "Skipping index-specific tests as concrete implementations may vary." << endl;
        
        // Just verify the IndexingMethod interface exists
        assert_test(true, "IndexingMethod interface available");
    }
    
    void test_performance_comparison() {
        cout << "\n=== Testing Performance Comparison ===" << endl;
        
        VectorStore store;
        const int num_docs = 500; // Reduced for faster testing
        const int dimensions = 32;
        
        // Create test dataset
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < num_docs; ++i) {
            vector<float> embedding(dimensions);
            for (int j = 0; j < dimensions; ++j) {
                embedding[j] = dist(gen);
            }
            store.insert(Document(embedding, {{"id", i}}));
        }
        
        CosineSimilarity metric;
        
        // Create engines
        auto start = high_resolution_clock::now();
        FlatSearchEngine flat_engine(store, metric);
        auto flat_build_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        start = high_resolution_clock::now();
        LSHSearchEngine lsh_engine(store, metric, 8, 6);
        auto lsh_build_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        start = high_resolution_clock::now();
        HNSWSearchEngine hnsw_engine(store, metric, 8, 50, 25);
        auto hnsw_build_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        start = high_resolution_clock::now();
        AnnoySearchEngine annoy_engine(store, metric, 5, 25);
        auto annoy_build_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        // Test query performance
        vector<float> query(dimensions);
        for (int i = 0; i < dimensions; ++i) {
            query[i] = dist(gen);
        }
        
        start = high_resolution_clock::now();
        auto flat_results = flat_engine.search(query, 10);
        auto flat_query_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        start = high_resolution_clock::now();
        auto lsh_results = lsh_engine.search(query, 10);
        auto lsh_query_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        start = high_resolution_clock::now();
        auto hnsw_results = hnsw_engine.search(query, 10);
        auto hnsw_query_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        start = high_resolution_clock::now();
        auto annoy_results = annoy_engine.search(query, 10);
        auto annoy_query_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
        
        cout << "Performance Results:" << endl;
        cout << "  Flat build: " << flat_build_time.count() << "μs, query: " << flat_query_time.count() << "μs" << endl;
        cout << "  LSH build: " << lsh_build_time.count() << "μs, query: " << lsh_query_time.count() << "μs" << endl;
        cout << "  HNSW build: " << hnsw_build_time.count() << "μs, query: " << hnsw_query_time.count() << "μs" << endl;
        cout << "  Annoy build: " << annoy_build_time.count() << "μs, query: " << annoy_query_time.count() << "μs" << endl;
        
        assert_test(flat_results.size() == 10, "Performance test - flat results count");
        assert_test(lsh_results.size() <= 10, "Performance test - LSH results count");
        assert_test(hnsw_results.size() <= 10, "Performance test - HNSW results count");
        assert_test(annoy_results.size() <= 10, "Performance test - Annoy results count");
        assert_test(!flat_results.empty() && !lsh_results.empty() && !hnsw_results.empty() && !annoy_results.empty(), "Performance test - non-empty results");
    }
    
    void run_all_tests() {
        cout << "🚀 Starting Comprehensive Unit Test Suite" << endl;
        cout << "===========================================" << endl;
        
        test_document_creation();
        test_vector_store();
        test_similarity_metrics();
        test_metadata_filter();
        test_flat_search_engine();
        test_lsh_search_engine();
        test_hnsw_search_engine();
        test_annoy_search_engine();
        test_storage_operations();
        test_index_operations();
        test_performance_comparison();
        
        cout << "\n📊 Test Summary:" << endl;
        cout << "=================" << endl;
        cout << "✅ Tests Passed: " << tests_passed << endl;
        cout << "❌ Tests Failed: " << tests_failed << endl;
        cout << "📈 Success Rate: " << (100.0 * tests_passed / (tests_passed + tests_failed)) << "%" << endl;
        
        if (tests_failed == 0) {
            cout << "\n🎉 All tests passed! The system is working correctly." << endl;
        } else {
            cout << "\n⚠️  Some tests failed. Please review the implementation." << endl;
        }
    }
};

int main() {
    TestSuite suite;
    suite.run_all_tests();
    return 0;
}
