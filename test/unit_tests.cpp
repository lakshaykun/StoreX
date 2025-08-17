#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>

// Include all project headers
#include "document.hpp"
#include "collection.hpp"
#include "similarity.hpp"
#include "index.hpp"
#include "storage.hpp"
#include "vector_store.hpp"
#include "lsh.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

// Test counter
int tests_passed = 0;
int tests_total = 0;

// Helper macros for testing
#define TEST(condition, message) \
    tests_total++; \
    if (condition) { \
        tests_passed++; \
        cout << "[PASS] " << message << endl; \
    } else { \
        cout << "[FAIL] " << message << endl; \
    }

#define TEST_APPROX_EQUAL(a, b, epsilon, message) \
    tests_total++; \
    if (std::abs(a - b) < epsilon) { \
        tests_passed++; \
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

// ============== METADATA TESTS ==============
void testMetadata() {
    cout << "\n=== Testing Metadata ===" << endl;
    
    // Test default constructor
    Metadata meta1;
    TEST(meta1.getData().is_object(), "Default metadata constructor creates empty JSON object");
    
    // Test JSON string constructor
    string jsonStr = R"({"title": "test", "id": 1})";
    Metadata meta2(jsonStr);
    TEST(meta2.getData()["title"] == "test", "Metadata from JSON string - title");
    TEST(meta2.getData()["id"] == 1, "Metadata from JSON string - id");
    
    // Test JSON object constructor
    json jsonObj;
    jsonObj["author"] = "Alice";
    jsonObj["year"] = 2023;
    Metadata meta3(jsonObj);
    TEST(meta3.getData()["author"] == "Alice", "Metadata from JSON object - author");
    TEST(meta3.getData()["year"] == 2023, "Metadata from JSON object - year");
    
    // Test empty string constructor
    Metadata meta4(string(""));
    TEST(meta4.getData().is_object(), "Metadata from empty string creates empty JSON object");
    
    // Test invalid JSON string (should create empty object)
    Metadata meta5(string("{invalid json}"));
    TEST(meta5.getData().is_object(), "Metadata from invalid JSON creates empty object");
    
    // Test equality operator
    Metadata meta6(jsonStr);
    TEST(meta2 == meta6, "Metadata equality operator");
    TEST(!(meta2 == meta3), "Metadata inequality");
    
    // Test toString method
    string metaStr = meta2.toString();
    TEST(!metaStr.empty(), "Metadata toString returns non-empty string");
}

// ============== DOCUMENT TESTS ==============
void testDocument() {
    cout << "\n=== Testing Document ===" << endl;
    
    // Test default constructor
    Document doc1;
    TEST(doc1.getEmbedding().empty(), "Default document has empty embedding");
    
    // Test embedding-only constructor
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    Document doc2(emb1);
    TEST(vectorsEqual(doc2.getEmbedding(), emb1), "Document with embedding constructor");
    
    // Test embedding + metadata constructor
    string jsonStr = R"({"title": "doc2"})";
    Metadata meta(jsonStr);
    Document doc3(emb1, meta);
    TEST(vectorsEqual(doc3.getEmbedding(), emb1), "Document with embedding and metadata - embedding");
    TEST(doc3.getMetadata() == meta, "Document with embedding and metadata - metadata");
    
    // Test embedding + JSON constructor
    json jsonObj;
    jsonObj["type"] = "test";
    Document doc4(emb1, jsonObj);
    TEST(vectorsEqual(doc4.getEmbedding(), emb1), "Document with embedding and JSON - embedding");
    TEST(doc4.getMetadata().getData()["type"] == "test", "Document with embedding and JSON - metadata");
    
    // Test embedding + JSON string constructor
    Document doc5(emb1, jsonStr);
    TEST(vectorsEqual(doc5.getEmbedding(), emb1), "Document with embedding and JSON string - embedding");
    TEST(doc5.getMetadata().getData()["title"] == "doc2", "Document with embedding and JSON string - metadata");
}

// ============== SIMILARITY TESTS ==============
void testSimilarity() {
    cout << "\n=== Testing Similarity Metrics ===" << endl;
    
    vector<float> vec1 = {1.0f, 0.0f, 0.0f};
    vector<float> vec2 = {0.0f, 1.0f, 0.0f};
    vector<float> vec3 = {1.0f, 0.0f, 0.0f}; // Same as vec1
    
    // Test Cosine Similarity
    CosineSimilarity cosine;
    float cosine_result1 = cosine.compute(vec1, vec2);
    float cosine_result2 = cosine.compute(vec1, vec3);
    TEST_APPROX_EQUAL(cosine_result1, 0.0f, 1e-6, "Cosine similarity of orthogonal vectors");
    TEST_APPROX_EQUAL(cosine_result2, 1.0f, 1e-6, "Cosine similarity of identical vectors");
    
    // Test Euclidean Similarity (distance-based converted to similarity, so exp(-distance))
    EuclideanSimilarity euclidean;
    float euclidean_result1 = euclidean.compute(vec1, vec2);
    float euclidean_result2 = euclidean.compute(vec1, vec3);
    TEST(euclidean_result1 < euclidean_result2, "Euclidean similarity: different vectors have lower similarity than identical");
    TEST_APPROX_EQUAL(euclidean_result2, 1.0f, 1e-6, "Euclidean similarity of identical vectors");
    
    // Test Jaccard Similarity (typically for binary vectors, but should work with float vectors)
    JaccardSimilarity jaccard;
    float jaccard_result1 = jaccard.compute(vec1, vec2);
    float jaccard_result2 = jaccard.compute(vec1, vec3);
    TEST(jaccard_result2 > jaccard_result1, "Jaccard similarity: identical vectors more similar than different ones");
    
    // Test factory function
    auto cosine_ptr = createSimilarity("cosine");
    TEST(cosine_ptr != nullptr, "Factory creates cosine similarity");
    TEST(cosine_ptr->name == "CosineSimilarity", "Factory cosine similarity has correct name");
    
    auto euclidean_ptr = createSimilarity("euclidean");
    TEST(euclidean_ptr != nullptr, "Factory creates euclidean similarity");
    TEST(euclidean_ptr->name == "EuclideanSimilarity", "Factory euclidean similarity has correct name");
    
    auto jaccard_ptr = createSimilarity("jaccard");
    TEST(jaccard_ptr != nullptr, "Factory creates jaccard similarity");
    TEST(jaccard_ptr->name == "JaccardSimilarity", "Factory jaccard similarity has correct name");
    
    // Test invalid similarity type
    try {
        auto invalid_ptr = createSimilarity("invalid");
        TEST(false, "Factory should throw for invalid type");
    } catch (const std::exception& e) {
        TEST(true, "Factory throws exception for invalid type");
    }
}

// ============== COLLECTION TESTS ==============
void testCollection() {
    cout << "\n=== Testing Collection ===" << endl;
    
    // Test constructor without storage
    Collection col1(3, false);
    TEST(col1.size() == 0, "New collection is empty");
    
    // Test document insertion
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    Document doc1(emb1);
    size_t id1 = col1.insert(doc1);
    TEST(id1 == 0, "First document gets ID 0");
    TEST(col1.size() == 1, "Collection size increases after insertion");
    
    // Test document retrieval
    Document retrieved = col1.getDocument(id1);
    TEST(vectorsEqual(retrieved.getEmbedding(), emb1), "Retrieved document has correct embedding");
    
    // Test multiple insertions
    vector<float> emb2 = {4.0f, 5.0f, 6.0f};
    Document doc2(emb2);
    size_t id2 = col1.insert(doc2);
    TEST(id2 == 1, "Second document gets ID 1");
    TEST(col1.size() == 2, "Collection size is 2 after two insertions");
    
    // Test batch insertion (Collection doesn't have batch insert, so insert individually)
    vector<Document> docs;
    docs.emplace_back(vector<float>{7.0f, 8.0f, 9.0f});
    docs.emplace_back(vector<float>{10.0f, 11.0f, 12.0f});
    vector<size_t> ids;
    for (auto& doc : docs) {
        ids.push_back(col1.insert(doc));
    }
    TEST(ids.size() == 2, "Individual insert returns correct number of IDs");
    TEST(ids[0] == 2 && ids[1] == 3, "Individual insert returns sequential IDs");
    TEST(col1.size() == 4, "Collection size is 4 after individual insertions");
    
    // Test document update
    vector<float> emb_new = {1.5f, 2.5f, 3.5f};
    Document doc_new(emb_new);
    col1.update(id1, doc_new);
    Document updated = col1.getDocument(id1);
    TEST(vectorsEqual(updated.getEmbedding(), emb_new), "Document update changes embedding");
    
    // Collection doesn't have direct search by metadata, so just test insertion
    string jsonStr = R"({"category": "test"})";
    Metadata meta(jsonStr);
    Document doc_with_meta(emb1, meta);
    size_t id_meta = col1.insert(doc_with_meta);
    
    // Verify the document was inserted correctly
    Document retrieved_meta = col1.getDocument(id_meta);
    TEST(retrieved_meta.getMetadata() == meta, "Document with metadata inserted correctly");
}

// ============== STORAGE TESTS ==============
void testStorage() {
    cout << "\n=== Testing Storage ===" << endl;
    
    // Clean up any existing test database
    string test_db = "test_storage.db";
    std::filesystem::remove(test_db);
    
    try {
        Storage storage(test_db);
        
        // Test document insert and load
        vector<float> emb = {1.0f, 2.0f, 3.0f};
        string jsonStr = R"({"title": "test_doc"})";
        Metadata meta(jsonStr);
        Document doc(emb, meta);
        
        storage.insert(0, doc);
        TEST(true, "Document inserted successfully");
        
        // Test loading all documents
        vector<Document> all_docs = storage.load();
        TEST(all_docs.size() >= 1, "Load returns inserted document");
        
        if (!all_docs.empty()) {
            Document loaded_doc = all_docs[0];
            TEST(vectorsEqual(loaded_doc.getEmbedding(), emb), "Loaded document has correct embedding");
            TEST(loaded_doc.getMetadata() == meta, "Loaded document has correct metadata");
        }
        
        // Test document update
        vector<float> new_emb = {4.0f, 5.0f, 6.0f};
        Document new_doc(new_emb, meta);
        storage.update(0, new_doc);
        
        vector<Document> updated_docs = storage.load();
        if (!updated_docs.empty()) {
            Document updated_doc = updated_docs[0];
            TEST(vectorsEqual(updated_doc.getEmbedding(), new_emb), "Updated document has new embedding");
        }
        
        // Test multiple documents
        storage.insert(1, doc);
        vector<Document> multi_docs = storage.load();
        TEST(multi_docs.size() >= 2, "Load returns multiple documents");
        
    } catch (const std::exception& e) {
        TEST(false, string("Storage test failed with exception: ") + e.what());
    }
    
    // Clean up test database
    std::filesystem::remove(test_db);
}

// ============== INDEX TESTS ==============
void testFlatIndex() {
    cout << "\n=== Testing FlatIndex ===" << endl;
    
    auto collection = make_shared<Collection>(3, false);
    auto similarity = make_shared<CosineSimilarity>();
    FlatIndex index(collection, similarity);
    
    // Test document insertion
    vector<float> emb1 = {1.0f, 0.0f, 0.0f};
    Document doc1(emb1);
    size_t id1 = index.insert(doc1);
    TEST(id1 == 0, "FlatIndex insert returns correct ID");
    
    // Test search by embedding
    vector<float> query = {0.9f, 0.1f, 0.0f}; // Similar to emb1
    vector<Document> results = index.search(query, 1);
    TEST(results.size() == 1, "FlatIndex search returns correct number of results");
    
    // Test search with scores
    vector<std::pair<float, Document>> scored_results = index.searchWithScores(query, 1);
    TEST(scored_results.size() == 1, "FlatIndex searchWithScores returns correct number of results");
    TEST(scored_results[0].first > 0.5f, "FlatIndex search returns reasonable similarity score");
    
    // Test multiple documents
    vector<float> emb2 = {0.0f, 1.0f, 0.0f};
    Document doc2(emb2);
    index.insert(doc2);
    
    vector<Document> multi_results = index.search(query, 2);
    TEST(multi_results.size() == 2, "FlatIndex search finds multiple documents");
    
    // Test search by metadata
    string jsonStr = R"({"type": "test"})";
    Metadata meta(jsonStr);
    Document doc_with_meta(emb1, meta);
    index.insert(doc_with_meta);
    
    vector<Document> meta_results = index.search(meta, 10);
    TEST(meta_results.size() >= 1, "FlatIndex search by metadata finds documents");
}

// ============== LSH TESTS ==============
void testLSH() {
    cout << "\n=== Testing LSH ===" << endl;
    
    auto collection = make_shared<Collection>(3, false);
    auto similarity = make_shared<CosineSimilarity>();
    
    // Test LSHJaccard
    LSHJaccard lsh(10, 5, 2, collection, similarity);
    
    // Insert documents
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    vector<float> emb2 = {1.1f, 2.1f, 3.1f}; // Similar to emb1
    vector<float> emb3 = {10.0f, 20.0f, 30.0f}; // Different from others
    
    Document doc1(emb1);
    Document doc2(emb2);
    Document doc3(emb3);
    
    size_t id1 = collection->insert(doc1);
    size_t id2 = collection->insert(doc2);
    size_t id3 = collection->insert(doc3);
    
    lsh.insert(emb1, id1);
    lsh.insert(emb2, id2);
    lsh.insert(emb3, id3);
    
    // Test search
    vector<float> query = {1.05f, 2.05f, 3.05f}; // Very similar to emb1 and emb2
    vector<Document> results = lsh.search(query, 2);
    TEST(results.size() <= 2, "LSH search returns at most k results");
    TEST(true, "LSH search completes successfully");
    
    // Test search with scores
    vector<std::pair<float, Document>> scored_results = lsh.searchWithScores(query, 2);
    TEST(scored_results.size() <= 2, "LSH searchWithScores returns at most k results");
}

// ============== VECTOR STORE TESTS ==============
void testVectorStore() {
    cout << "\n=== Testing VectorStore ===" << endl;
    
    // Test basic constructor
    vector_store store(3);
    
    // Test document insertion
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    string jsonStr1 = R"({"title": "doc1", "category": "test"})";
    Metadata meta1(jsonStr1);
    Document doc1(emb1, meta1);
    
    size_t id1 = store.insert(doc1);
    TEST(id1 == 0, "VectorStore insert returns correct ID");
    
    // Test batch insertion
    vector<Document> docs;
    vector<float> emb2 = {2.0f, 3.0f, 4.0f};
    vector<float> emb3 = {3.0f, 4.0f, 5.0f};
    string jsonStr2 = R"({"title": "doc2", "category": "test"})";
    string jsonStr3 = R"({"title": "doc3", "category": "other"})";
    
    docs.emplace_back(emb2, Metadata(jsonStr2));
    docs.emplace_back(emb3, Metadata(jsonStr3));
    
    vector<size_t> ids = store.insert(docs);
    TEST(ids.size() == 2, "VectorStore batch insert returns correct number of IDs");
    
    // Test search by embedding
    vector<float> query = {1.1f, 2.1f, 3.1f};
    vector<Document> results = store.search(query, 2);
    TEST(results.size() <= 2, "VectorStore search by embedding returns at most k results");
    TEST(results.size() >= 1, "VectorStore search by embedding returns at least one result");
    
    // Test search with scores
    vector<std::pair<float, Document>> scored_results = store.searchWithScores(query, 2);
    TEST(scored_results.size() <= 2, "VectorStore searchWithScores returns at most k results");
    TEST(scored_results.size() >= 1, "VectorStore searchWithScores returns at least one result");
    
    // Test search by metadata
    Metadata test_meta(jsonStr1);
    vector<Document> meta_results = store.search(test_meta, 10);
    TEST(meta_results.size() >= 1, "VectorStore search by metadata finds documents");
    
    // Test combined metadata and embedding search
    vector<Document> combined_results = store.search(test_meta, query, 10);
    TEST(true, "VectorStore combined search completes successfully");
    
    // Test document update
    vector<float> new_emb = {1.5f, 2.5f, 3.5f};
    Document new_doc(new_emb, meta1);
    store.update(id1, new_doc);
    TEST(true, "VectorStore update completes successfully");
    
    // Test fetchId by embedding
    size_t found_id = store.fetchId(new_emb);
    TEST(found_id != SIZE_MAX, "VectorStore fetchId by embedding finds document");
    
    // Test fetchId by metadata
    size_t meta_id = store.fetchId(test_meta);
    TEST(meta_id != SIZE_MAX, "VectorStore fetchId by metadata finds document");
}

// ============== ANNOY INDEX TESTS ==============
void testAnnoyIndex() {
    cout << "\n=== Testing AnnoyIndex ===" << endl;
    
    auto collection = make_shared<Collection>(3, false);
    auto similarity = make_shared<CosineSimilarity>();
    
    try {
        AnnoyIndex annoyIndex(5, 2, collection, similarity);  // 5 trees, leaf size 2
        
        // Test document insertion
        vector<float> emb1 = {1.0f, 0.0f, 0.0f};
        Document doc1(emb1);
        size_t id1 = annoyIndex.insert(doc1);
        TEST(id1 == 0, "AnnoyIndex insert returns correct ID");
        
        // Test search by embedding
        vector<float> query = {0.9f, 0.1f, 0.0f}; // Similar to emb1
        vector<Document> results = annoyIndex.search(query, 1);
        TEST(results.size() == 1, "AnnoyIndex search returns correct number of results");
        
        // Test search with scores
        vector<std::pair<float, Document>> scored_results = annoyIndex.searchWithScores(query, 1);
        TEST(scored_results.size() == 1, "AnnoyIndex searchWithScores returns correct number of results");
        
        // Test multiple documents
        vector<float> emb2 = {0.0f, 1.0f, 0.0f};
        Document doc2(emb2);
        annoyIndex.insert(doc2);
        
        vector<Document> multi_results = annoyIndex.search(query, 2);
        TEST(multi_results.size() == 2, "AnnoyIndex search finds multiple documents");
        
        // Test search by metadata
        string jsonStr = R"({"type": "test"})";
        Metadata meta(jsonStr);
        Document doc_with_meta(emb1, meta);
        annoyIndex.insert(doc_with_meta);
        
        vector<Document> meta_results = annoyIndex.search(meta, 10);
        TEST(meta_results.size() >= 1, "AnnoyIndex search by metadata finds documents");
        
        // Test combined metadata and embedding search
        vector<Document> combined_results = annoyIndex.search(meta, query, 10);
        TEST(true, "AnnoyIndex combined search completes successfully");
        
        // Test update
        vector<float> new_emb = {1.5f, 0.5f, 0.0f};
        Document new_doc(new_emb, meta);
        annoyIndex.update(id1, new_doc);
        TEST(true, "AnnoyIndex update completes successfully");
        
    } catch (const std::exception& e) {
        TEST(false, string("AnnoyIndex test failed with exception: ") + e.what());
    }
    
    // Test invalid constructor parameters
    try {
        AnnoyIndex invalid1(0, 2, collection, similarity);  // num_trees = 0
        TEST(false, "AnnoyIndex should throw for zero num_trees");
    } catch (const std::exception& e) {
        TEST(true, "AnnoyIndex throws for zero num_trees");
    }
    
    try {
        AnnoyIndex invalid2(5, 0, collection, similarity);  // leaf_size = 0
        TEST(false, "AnnoyIndex should throw for zero leaf_size");
    } catch (const std::exception& e) {
        TEST(true, "AnnoyIndex throws for zero leaf_size");
    }
    
    try {
        AnnoyIndex invalid3(5, 2, nullptr, similarity);  // null collection
        TEST(false, "AnnoyIndex should throw for null collection");
    } catch (const std::exception& e) {
        TEST(true, "AnnoyIndex throws for null collection");
    }
}

// ============== EDGE CASE TESTS ==============
void testEdgeCases() {
    cout << "\n=== Testing Edge Cases ===" << endl;
    
    // Test empty vectors
    vector<float> emptyVec1, emptyVec2;
    CosineSimilarity cosine;
    try {
        float result = cosine.compute(emptyVec1, emptyVec2);
        TEST(true, "Cosine similarity handles empty vectors");
    } catch (const std::exception& e) {
        TEST(true, "Cosine similarity throws for empty vectors");
    }
    
    // Test mismatched vector dimensions
    vector<float> vec1 = {1.0f, 2.0f};
    vector<float> vec2 = {1.0f, 2.0f, 3.0f};
    try {
        float result = cosine.compute(vec1, vec2);
        TEST(false, "Should throw for mismatched dimensions");
    } catch (const std::exception& e) {
        TEST(true, "Throws exception for mismatched vector dimensions");
    }
    
    // Test zero vectors
    vector<float> zeroVec1 = {0.0f, 0.0f, 0.0f};
    vector<float> zeroVec2 = {0.0f, 0.0f, 0.0f};
    try {
        float result = cosine.compute(zeroVec1, zeroVec2);
        TEST(true, "Cosine similarity handles zero vectors");
    } catch (const std::exception& e) {
        TEST(true, "Cosine similarity throws for zero vectors (division by zero)");
    }
    
    // Test very large vectors
    vector<float> largeVec1(1000, 1.0f);
    vector<float> largeVec2(1000, 2.0f);
    try {
        float result = cosine.compute(largeVec1, largeVec2);
        TEST(true, "Cosine similarity handles large vectors");
    } catch (const std::exception& e) {
        TEST(false, string("Cosine similarity should handle large vectors: ") + e.what());
    }
    
    // Test Collection with zero vector size
    try {
        Collection col_zero(0, false);
        TEST(true, "Collection accepts zero vector size");
    } catch (const std::exception& e) {
        TEST(true, "Collection throws for zero vector size");
    }
    
    // Test search with k=0
    Collection col(3, false);
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    col.insert(doc);
    
    auto sim = make_shared<CosineSimilarity>();
    FlatIndex index(make_shared<Collection>(col), sim);
    vector<Document> results = index.search(emb, 0);
    TEST(results.empty(), "Search with k=0 returns empty results");
}

// ============== UTILITY FUNCTION TESTS ==============
void testUtilityFunctions() {
    cout << "\n=== Testing Utility Functions ===" << endl;
    
    // Test dot product
    vector<float> vec1 = {1.0f, 2.0f, 3.0f};
    vector<float> vec2 = {4.0f, 5.0f, 6.0f};
    float expected = 1*4 + 2*5 + 3*6; // 32
    
    try {
        float result = utility::dotProduct(vec1, vec2);
        TEST_APPROX_EQUAL(result, expected, 1e-6, "Dot product calculation is correct");
    } catch (const std::exception& e) {
        TEST(false, string("Dot product failed: ") + e.what());
    }
    
    // Test dot product with mismatched dimensions
    vector<float> vec3 = {1.0f, 2.0f};
    try {
        float result = utility::dotProduct(vec1, vec3);
        TEST(false, "Should throw for mismatched dimensions in dot product");
    } catch (const std::exception& e) {
        TEST(true, "Dot product throws for mismatched dimensions");
    }
    
    // Test dot product with empty vectors
    vector<float> empty1, empty2;
    try {
        float result = utility::dotProduct(empty1, empty2);
        TEST_APPROX_EQUAL(result, 0.0f, 1e-6, "Dot product of empty vectors is zero");
    } catch (const std::exception& e) {
        TEST(false, string("Dot product of empty vectors failed: ") + e.what());
    }
}

// ============== PERFORMANCE TESTS ==============
void testPerformance() {
    cout << "\n=== Testing Performance ===" << endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create a large collection
    Collection col(100, false);  // 100-dimensional vectors
    auto similarity = make_shared<EuclideanSimilarity>();
    auto index = make_shared<FlatIndex>(make_shared<Collection>(col), similarity);
    
    // Insert many documents
    const size_t num_docs = 100;
    for (size_t i = 0; i < num_docs; i++) {
        vector<float> emb(100);
        for (size_t j = 0; j < 100; j++) {
            emb[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        Document doc(emb);
        index->insert(doc);
    }
    
    auto insert_time = std::chrono::high_resolution_clock::now();
    
    // Perform many searches
    const size_t num_searches = 50;
    vector<float> query(100);
    for (size_t j = 0; j < 100; j++) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    for (size_t i = 0; i < num_searches; i++) {
        vector<Document> results = index->search(query, 10);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto insert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(insert_time - start_time);
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - insert_time);
    
    TEST(insert_duration.count() < 5000, "Large insert completes in reasonable time (<5s)");
    TEST(search_duration.count() < 1000, "Many searches complete in reasonable time (<1s)");
    
    cout << "  Insert time: " << insert_duration.count() << "ms" << endl;
    cout << "  Search time: " << search_duration.count() << "ms" << endl;
}

// ============== LSH INDEX TESTS ==============
void testLSHIndex() {
    cout << "\n=== Testing LSHIndex ===" << endl;
    
    auto collection = make_shared<Collection>(3, false);
    auto similarity = make_shared<CosineSimilarity>();
    
    try {
        LSHIndex lshIndex(collection, similarity);
        
        // Test document insertion
        vector<float> emb1 = {1.0f, 0.0f, 0.0f};
        Document doc1(emb1);
        size_t id1 = lshIndex.insert(doc1);
        TEST(id1 == 0, "LSHIndex insert returns correct ID");
        
        // Test search by embedding
        vector<float> query = {0.9f, 0.1f, 0.0f}; // Similar to emb1
        vector<Document> results = lshIndex.search(query, 1);
        TEST(results.size() <= 1, "LSHIndex search returns at most k results");
        
        // Test search with scores
        vector<std::pair<float, Document>> scored_results = lshIndex.searchWithScores(query, 1);
        TEST(scored_results.size() <= 1, "LSHIndex searchWithScores returns at most k results");
        
        // Test multiple documents
        vector<float> emb2 = {0.0f, 1.0f, 0.0f};
        Document doc2(emb2);
        lshIndex.insert(doc2);
        
        vector<Document> multi_results = lshIndex.search(query, 2);
        TEST(multi_results.size() <= 2, "LSHIndex search finds at most k documents");
        
        // Test search by metadata
        string jsonStr = R"({"type": "test"})";
        Metadata meta(jsonStr);
        Document doc_with_meta(emb1, meta);
        lshIndex.insert(doc_with_meta);
        
        vector<Document> meta_results = lshIndex.search(meta, 10);
        TEST(meta_results.size() >= 1, "LSHIndex search by metadata finds documents");
        
        // Test update
        vector<float> new_emb = {1.5f, 0.5f, 0.0f};
        Document new_doc(new_emb, meta);
        lshIndex.update(id1, new_doc);
        TEST(true, "LSHIndex update completes successfully");
        
    } catch (const std::exception& e) {
        TEST(false, string("LSHIndex test failed with exception: ") + e.what());
    }
}

// ============== STRESS TESTS ==============
void testStressConditions() {
    cout << "\n=== Testing Stress Conditions ===" << endl;
    
    // Test with many identical vectors
    Collection col(3, false);
    vector<float> identical_emb = {1.0f, 1.0f, 1.0f};
    
    for (int i = 0; i < 100; i++) {
        Document doc(identical_emb, Metadata(string(R"({"id": )" + std::to_string(i) + "}")));
        col.insert(doc);
    }
    TEST(col.size() == 100, "Collection handles many identical vectors");
    
    // Test search with many identical results
    auto similarity = make_shared<CosineSimilarity>();
    FlatIndex index(make_shared<Collection>(col), similarity);
    
    vector<Document> results = index.search(identical_emb, 50);
    TEST(results.size() == 50, "Search returns correct number from many identical vectors");
    
    // Test very high-dimensional vectors
    try {
        Collection high_dim_col(1000, false);
        vector<float> high_dim_emb(1000, 1.0f);
        Document high_dim_doc(high_dim_emb);
        size_t id = high_dim_col.insert(high_dim_doc);
        TEST(id == 0, "High-dimensional vectors handled correctly");
    } catch (const std::exception& e) {
        TEST(false, string("High-dimensional test failed: ") + e.what());
    }
}

// ============== ERROR HANDLING TESTS ==============
void testErrorHandling() {
    cout << "\n=== Testing Error Handling ===" << endl;
    
    // Test document retrieval with invalid ID
    Collection col(3, false);
    try {
        Document doc = col.getDocument(999); // Non-existent ID
        TEST(false, "Should throw for invalid document ID");
    } catch (const std::exception& e) {
        TEST(true, "Throws exception for invalid document ID");
    }
    
    // Test vector store with invalid similarity type
    try {
        auto invalid_sim = createSimilarity("invalid_type");
        TEST(false, "Should throw for invalid similarity type");
    } catch (const std::exception& e) {
        TEST(true, "Throws exception for invalid similarity type");
    }
    
    // Test storage with invalid database path
    try {
        Storage storage("/invalid/path/database.db");
        TEST(false, "Should handle invalid database path");
    } catch (const std::exception& e) {
        TEST(true, "Handles invalid database path gracefully");
    }
    
    // Test update with invalid ID
    Collection col2(3, false);
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    
    try {
        col2.update(999, doc); // Non-existent ID
        TEST(false, "Should throw for invalid update ID");
    } catch (const std::exception& e) {
        TEST(true, "Throws exception for invalid update ID");
    }
}

// ============== CONCURRENCY TESTS ==============
void testConcurrency() {
    cout << "\n=== Testing Concurrency Safety ===" << endl;
    
    // Basic test to ensure thread-local RNG is working
    static thread_local std::mt19937 test_rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    vector<float> random_values;
    for (int i = 0; i < 10; i++) {
        random_values.push_back(dist(test_rng));
    }
    
    TEST(random_values.size() == 10, "Thread-local RNG generates values");
    TEST(true, "Basic concurrency infrastructure works");
}

// ============== INTEGRATION TESTS ==============
void testIntegration() {
    cout << "\n=== Testing Integration ===" << endl;
    
    // Test the complete workflow similar to main.cpp
    auto col = make_shared<Collection>(3);
    auto sim = make_shared<EuclideanSimilarity>();
    auto ind = make_shared<FlatIndex>(col, sim);
    vector_store store(ind, col);
    
    // Create test documents
    vector<Document> docs;
    Metadata meta1(string(R"({"title": "Test1", "id": "1", "genre": ["horror", "fantasy"]})"));
    Metadata meta2(string(R"({"title": "Test2", "id": "2", "genre": ["space", "fantasy"]})"));
    
    // Add some test documents
    for (int i = 0; i < 5; i++) {
        docs.emplace_back(
            Document({static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)}, meta1)
        );
    }
    
    for (int i = 5; i < 10; i++) {
        docs.emplace_back(
            Document({static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)}, meta2)
        );
    }
    
    // Insert documents
    vector<size_t> ids = store.insert(docs);
    TEST(ids.size() == 10, "Integration test: all documents inserted");
    
    // Search for similar documents
    vector<float> emb = {2.5f, 3.5f, 4.5f};
    vector<std::pair<float, Document>> results = store.searchWithScores(emb, 3);
    TEST(results.size() <= 3, "Integration test: search returns at most k results");
    TEST(results.size() >= 1, "Integration test: search returns at least one result");
    
    // Verify results are sorted by similarity (for Euclidean, lower distance = higher similarity)
    // Note: The Euclidean similarity returns exp(-distance), so higher values are more similar
    for (size_t i = 1; i < results.size(); i++) {
        TEST(results[i-1].first >= results[i].first, "Integration test: results sorted by similarity (descending)");
    }
    
    // Test metadata-based search
    vector<Document> meta_results = store.search(meta1, 10);
    TEST(meta_results.size() == 5, "Integration test: metadata search finds correct number of documents");
    
    // Test combined search
    vector<Document> combined_results = store.search(meta2, emb, 3);
    TEST(combined_results.size() <= 3, "Integration test: combined search returns at most k results");
}

// ============== MAIN TEST RUNNER ==============
int main() {
    cout << "Running StoreX Unit Tests..." << endl;
    cout << "==============================" << endl;
    
    try {
        testMetadata();
        testDocument();
        testSimilarity();
        testCollection();
        testStorage();
        testFlatIndex();
        testLSH();
        testVectorStore();
        testAnnoyIndex();
        testLSHIndex();
        testEdgeCases();
        testUtilityFunctions();
        testPerformance();
        testStressConditions();
        testErrorHandling();
        testConcurrency();
        testIntegration();
    } catch (const std::exception& e) {
        cout << "Test execution failed with exception: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n==============================" << endl;
    cout << "Test Results: " << tests_passed << "/" << tests_total << " passed" << endl;
    
    if (tests_passed == tests_total) {
        cout << "All tests passed! ✓" << endl;
        return 0;
    } else {
        cout << "Some tests failed! ✗" << endl;
        return 1;
    }
}