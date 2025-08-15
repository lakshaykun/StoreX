#include "vector_store.hpp"
#include "document.hpp"
#include "collection.hpp"
#include "similarity.hpp"
#include "index.hpp"
#include "storage.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>

using std::cout;
using std::endl;
using std::abs;

// Helper function to compare vectors
bool vectors_equal(const vector<float>& v1, const vector<float>& v2) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); i++) {
        if (abs(v1[i] - v2[i]) > 1e-6f) return false;
    }
    return true;
}

// Test counter
int tests_passed = 0;
int tests_failed = 0;

// Helper macros for testing
#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        cout << "FAILED: " << __FUNCTION__ << " line " << __LINE__ << ": " << #condition << endl; \
        tests_failed++; \
        return false; \
    } else { \
        tests_passed++; \
    }

#define ASSERT_FALSE(condition) \
    if (condition) { \
        cout << "FAILED: " << __FUNCTION__ << " line " << __LINE__ << ": " << #condition << " should be false" << endl; \
        tests_failed++; \
        return false; \
    } else { \
        tests_passed++; \
    }

#define ASSERT_EQ(expected, actual) \
    if ((expected) != (actual)) { \
        cout << "FAILED: " << __FUNCTION__ << " line " << __LINE__ << ": Expected and actual values don't match" << endl; \
        tests_failed++; \
        return false; \
    } else { \
        tests_passed++; \
    }

#define ASSERT_NEAR(expected, actual, tolerance) \
    if (abs((expected) - (actual)) > (tolerance)) { \
        cout << "FAILED: " << __FUNCTION__ << " line " << __LINE__ << ": Expected " << (expected) << " but got " << (actual) << " (tolerance: " << (tolerance) << ")" << endl; \
        tests_failed++; \
        return false; \
    } else { \
        tests_passed++; \
    }

#define RUN_TEST(test_func) \
    cout << "Running " << #test_func << "... "; \
    if (test_func()) { \
        cout << "PASSED" << endl; \
    } else { \
        cout << "FAILED" << endl; \
    }

// =============================================================================
// METADATA TESTS
// =============================================================================

bool test_metadata_default_constructor() {
    Metadata meta;
    ASSERT_TRUE(meta.getData().is_object());
    ASSERT_TRUE(meta.getData().empty());
    return true;
}

bool test_metadata_json_constructor() {
    json j = {{"name", "test"}, {"id", 123}};
    Metadata meta(j);
    ASSERT_EQ(j, meta.getData());
    return true;
}

bool test_metadata_string_constructor() {
    string jsonStr = R"({"name": "test", "id": 123})";
    Metadata meta(jsonStr);
    ASSERT_EQ("test", meta.getData()["name"]);
    ASSERT_EQ(123, meta.getData()["id"]);
    return true;
}

bool test_metadata_invalid_json_string() {
    string invalidJson = "invalid json";
    Metadata meta(invalidJson);
    ASSERT_TRUE(meta.getData().is_object());
    ASSERT_TRUE(meta.getData().empty());
    return true;
}

bool test_metadata_empty_string() {
    string emptyStr = "";
    Metadata meta(emptyStr);
    ASSERT_TRUE(meta.getData().is_object());
    ASSERT_TRUE(meta.getData().empty());
    return true;
}

bool test_metadata_equality() {
    json j1 = {{"name", "test"}};
    json j2 = {{"name", "test"}};
    json j3 = {{"name", "different"}};
    
    Metadata meta1(j1);
    Metadata meta2(j2);
    Metadata meta3(j3);
    
    ASSERT_TRUE(meta1 == meta2);
    ASSERT_FALSE(meta1 == meta3);
    return true;
}

bool test_metadata_to_string() {
    json j = {{"name", "test"}, {"id", 123}};
    Metadata meta(j);
    string result = meta.toString();
    ASSERT_TRUE(result.find("test") != string::npos);
    ASSERT_TRUE(result.find("123") != string::npos);
    return true;
}

// =============================================================================
// DOCUMENT TESTS
// =============================================================================

bool test_document_default_constructor() {
    Document doc;
    ASSERT_TRUE(doc.getEmbedding().empty());
    ASSERT_TRUE(doc.getMetadata().getData().empty());
    ASSERT_TRUE(doc.isEmpty());
    return true;
}

bool test_document_embedding_constructor() {
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    ASSERT_TRUE(vectors_equal(emb, doc.getEmbedding()));
    ASSERT_TRUE(doc.getMetadata().getData().is_object());
    ASSERT_FALSE(doc.isEmpty());
    return true;
}

bool test_document_embedding_metadata_constructor() {
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    json j = {{"name", "test"}};
    Metadata meta(j);
    Document doc(emb, meta);
    
    ASSERT_TRUE(vectors_equal(emb, doc.getEmbedding()));
    ASSERT_TRUE(doc.getMetadata() == meta);
    ASSERT_FALSE(doc.isEmpty());
    return true;
}

bool test_document_setters() {
    Document doc;
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    json j = {{"name", "test"}};
    Metadata meta(j);
    
    doc.setEmbedding(emb);
    doc.setMetadata(meta);
    
    ASSERT_TRUE(vectors_equal(emb, doc.getEmbedding()));
    ASSERT_TRUE(doc.getMetadata() == meta);
    ASSERT_FALSE(doc.isEmpty());
    return true;
}

bool test_document_equality() {
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    vector<float> emb2 = {1.0f, 2.0f, 3.0f};
    vector<float> emb3 = {1.0f, 2.0f, 4.0f};
    
    json j1 = {{"name", "test"}};
    json j2 = {{"name", "test"}};
    json j3 = {{"name", "different"}};
    
    Document doc1(emb1, Metadata(j1));
    Document doc2(emb2, Metadata(j2));
    Document doc3(emb3, Metadata(j1));
    Document doc4(emb1, Metadata(j3));
    
    ASSERT_TRUE(doc1 == doc2);
    ASSERT_FALSE(doc1 == doc3);
    ASSERT_FALSE(doc1 == doc4);
    ASSERT_TRUE(doc1 != doc3);
    return true;
}

// =============================================================================
// SIMILARITY TESTS
// =============================================================================

bool test_cosine_similarity() {
    CosineSimilarity cosine;
    
    // Test identical vectors
    vector<float> v1 = {1.0f, 0.0f, 0.0f};
    vector<float> v2 = {1.0f, 0.0f, 0.0f};
    float result = cosine.compute(v1, v2);
    ASSERT_NEAR(1.0f, result, 1e-6f);
    
    // Test orthogonal vectors
    vector<float> v3 = {1.0f, 0.0f, 0.0f};
    vector<float> v4 = {0.0f, 1.0f, 0.0f};
    result = cosine.compute(v3, v4);
    ASSERT_NEAR(0.0f, result, 1e-6f);
    
    // Test opposite vectors
    vector<float> v5 = {1.0f, 0.0f, 0.0f};
    vector<float> v6 = {-1.0f, 0.0f, 0.0f};
    result = cosine.compute(v5, v6);
    ASSERT_NEAR(-1.0f, result, 1e-6f);
    
    return true;
}

bool test_euclidean_similarity() {
    EuclideanSimilarity euclidean;
    
    // Test identical vectors
    vector<float> v1 = {1.0f, 2.0f, 3.0f};
    vector<float> v2 = {1.0f, 2.0f, 3.0f};
    float result = euclidean.compute(v1, v2);
    ASSERT_NEAR(1.0f, result, 1e-6f); // exp(-0) = 1
    
    // Test different vectors
    vector<float> v3 = {0.0f, 0.0f, 0.0f};
    vector<float> v4 = {1.0f, 0.0f, 0.0f};
    result = euclidean.compute(v3, v4);
    ASSERT_NEAR(exp(-1.0f), result, 1e-6f);
    
    return true;
}

bool test_jaccard_similarity() {
    JaccardSimilarity jaccard;
    
    // Test identical vectors
    vector<float> v1 = {1.0f, 2.0f, 3.0f};
    vector<float> v2 = {1.0f, 2.0f, 3.0f};
    float result = jaccard.compute(v1, v2);
    ASSERT_NEAR(1.0f, result, 1e-6f);
    
    // Test disjoint vectors
    vector<float> v3 = {1.0f, 0.0f, 0.0f};
    vector<float> v4 = {0.0f, 1.0f, 0.0f};
    result = jaccard.compute(v3, v4);
    ASSERT_NEAR(0.0f, result, 1e-6f);
    
    return true;
}

bool test_similarity_factory() {
    auto cosine = createSimilarity("cosine");
    auto euclidean = createSimilarity("euclidean");
    auto jaccard = createSimilarity("jaccard");
    
    ASSERT_TRUE(cosine != nullptr);
    ASSERT_TRUE(euclidean != nullptr);
    ASSERT_TRUE(jaccard != nullptr);
    
    // Test invalid type
    try {
        auto invalid = createSimilarity("invalid");
        ASSERT_TRUE(false); // Should not reach here
    } catch (const invalid_argument& e) {
        ASSERT_TRUE(true); // Expected exception
    }
    
    return true;
}

bool test_similarity_edge_cases() {
    CosineSimilarity cosine;
    
    // Test empty vectors
    try {
        vector<float> empty1, empty2;
        cosine.compute(empty1, empty2);
        ASSERT_TRUE(false); // Should throw
    } catch (const invalid_argument& e) {
        ASSERT_TRUE(true);
    }
    
    // Test different sizes
    try {
        vector<float> v1 = {1.0f, 2.0f};
        vector<float> v2 = {1.0f, 2.0f, 3.0f};
        cosine.compute(v1, v2);
        ASSERT_TRUE(false); // Should throw
    } catch (const invalid_argument& e) {
        ASSERT_TRUE(true);
    }
    
    // Test zero vectors
    try {
        vector<float> zero1 = {0.0f, 0.0f, 0.0f};
        vector<float> zero2 = {0.0f, 0.0f, 0.0f};
        cosine.compute(zero1, zero2);
        ASSERT_TRUE(false); // Should throw
    } catch (const invalid_argument& e) {
        ASSERT_TRUE(true);
    }
    
    return true;
}

// =============================================================================
// COLLECTION TESTS
// =============================================================================

bool test_collection_default_constructor() {
    Collection collection;
    ASSERT_EQ(0, collection.size());
    ASSERT_TRUE(collection.getDocuments().empty());
    return true;
}

bool test_collection_insert() {
    Collection collection;
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    
    size_t id = collection.insert(doc);
    ASSERT_EQ(0, id);
    ASSERT_EQ(1, collection.size());
    ASSERT_TRUE(vectors_equal(emb, collection.getDocuments()[0].getEmbedding()));
    return true;
}

bool test_collection_update() {
    Collection collection;
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    vector<float> emb2 = {4.0f, 5.0f, 6.0f};
    Document doc1(emb1);
    Document doc2(emb2);
    
    size_t id = collection.insert(doc1);
    collection.update(id, doc2);
    
    ASSERT_EQ(1, collection.size());
    ASSERT_TRUE(vectors_equal(emb2, collection.getDocuments()[0].getEmbedding()));
    return true;
}

bool test_collection_update_out_of_range() {
    Collection collection;
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    
    try {
        collection.update(999, doc);
        ASSERT_TRUE(false); // Should throw
    } catch (const std::out_of_range& e) {
        ASSERT_TRUE(true);
    }
    return true;
}

bool test_collection_multiple_inserts() {
    Collection collection;
    vector<Document> docs;
    
    for (int i = 0; i < 5; i++) {
        vector<float> emb = {static_cast<float>(i), static_cast<float>(i+1)};
        Document doc(emb);
        docs.push_back(doc);
        size_t id = collection.insert(doc);
        ASSERT_EQ(i, id);
    }
    
    ASSERT_EQ(5, collection.size());
    for (size_t i = 0; i < 5; i++) {
        ASSERT_TRUE(vectors_equal(docs[i].getEmbedding(), collection.getDocuments()[i].getEmbedding()));
    }
    return true;
}

// =============================================================================
// STORAGE TESTS
// =============================================================================

bool test_storage_constructor() {
    // Clean up any existing test database
    string testDbPath = "test_db.db";
    if (std::filesystem::exists(testDbPath)) {
        std::filesystem::remove(testDbPath);
    }
    
    try {
        Storage storage(testDbPath);
        ASSERT_TRUE(std::filesystem::exists(testDbPath));
        
        // Clean up
        std::filesystem::remove(testDbPath);
    } catch (const std::exception& e) {
        cout << "Storage test failed with exception: " << e.what() << endl;
        return false;
    }
    return true;
}

bool test_storage_insert_and_load() {
    // Clean up any existing test database
    string testDbPath = "test_storage.db";
    if (std::filesystem::exists(testDbPath)) {
        std::filesystem::remove(testDbPath);
    }
    
    try {
        {
            Storage storage(testDbPath);
            vector<float> emb = {1.0f, 2.0f, 3.0f};
            json j = {{"name", "test_doc"}};
            Document doc(emb, Metadata(j));
            
            storage.insert(0, doc);
        } // Storage destructor closes database
        
        // Verify file exists
        ASSERT_TRUE(std::filesystem::exists(testDbPath));
        
        // Clean up
        std::filesystem::remove(testDbPath);
    } catch (const std::exception& e) {
        cout << "Storage insert test failed with exception: " << e.what() << endl;
        return false;
    }
    return true;
}

// =============================================================================
// FLAT INDEX TESTS
// =============================================================================

bool test_flat_index_constructor() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    
    FlatIndex index(collection, similarity);
    // If constructor completes without throwing, test passes
    return true;
}

bool test_flat_index_insert() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    
    size_t id = index.insert(doc);
    ASSERT_EQ(0, id);
    ASSERT_EQ(1, collection->size());
    return true;
}

bool test_flat_index_update() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    vector<float> emb1 = {1.0f, 2.0f, 3.0f};
    vector<float> emb2 = {4.0f, 5.0f, 6.0f};
    Document doc1(emb1);
    Document doc2(emb2);
    
    size_t id = index.insert(doc1);
    index.update(id, doc2);
    
    ASSERT_TRUE(vectors_equal(emb2, collection->getDocuments()[0].getEmbedding()));
    return true;
}

bool test_flat_index_search_by_metadata() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    // Insert documents with different metadata
    json j1 = {{"category", "A"}};
    json j2 = {{"category", "B"}};
    json j3 = {{"category", "A"}};
    
    vector<float> emb1 = {1.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f};
    vector<float> emb3 = {1.0f, 1.0f};
    
    Document doc1(emb1, Metadata(j1));
    Document doc2(emb2, Metadata(j2));
    Document doc3(emb3, Metadata(j3));
    
    index.insert(doc1);
    index.insert(doc2);
    index.insert(doc3);
    
    // Search for documents with category "A"
    Metadata searchMeta(j1);
    vector<Document> results = index.search(searchMeta, 10);
    
    ASSERT_EQ(2, results.size());
    ASSERT_TRUE(results[0].getMetadata() == searchMeta);
    ASSERT_TRUE(results[1].getMetadata() == searchMeta);
    return true;
}

bool test_flat_index_search_by_embedding() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    // Insert documents
    vector<float> emb1 = {1.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f};
    vector<float> emb3 = {0.7071f, 0.7071f}; // 45 degrees
    
    Document doc1(emb1);
    Document doc2(emb2);
    Document doc3(emb3);
    
    index.insert(doc1);
    index.insert(doc2);
    index.insert(doc3);
    
    // Search for most similar to emb1
    vector<Document> results = index.search(emb1, 2);
    
    ASSERT_EQ(2, results.size());
    // First result should be identical to query vector
    ASSERT_TRUE(vectors_equal(emb1, results[0].getEmbedding()));
    return true;
}

bool test_flat_index_search_with_scores() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    // Insert documents
    vector<float> emb1 = {1.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f};
    
    Document doc1(emb1);
    Document doc2(emb2);
    
    index.insert(doc1);
    index.insert(doc2);
    
    // Search with scores
    vector<std::pair<float, Document>> results = index.searchWithScores(emb1, 2);
    
    ASSERT_EQ(2, results.size());
    ASSERT_NEAR(1.0f, results[0].first, 1e-6f); // Perfect match
    ASSERT_NEAR(0.0f, results[1].first, 1e-6f); // Orthogonal
    return true;
}

bool test_flat_index_empty_collection_search() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    vector<float> queryEmb = {1.0f, 0.0f};
    vector<Document> results = index.search(queryEmb, 5);
    
    ASSERT_TRUE(results.empty());
    return true;
}

bool test_flat_index_search_k_zero() {
    auto collection = make_shared<Collection>();
    auto similarity = createSimilarity("cosine");
    FlatIndex index(collection, similarity);
    
    vector<float> emb = {1.0f, 0.0f};
    Document doc(emb);
    index.insert(doc);
    
    vector<Document> results = index.search(emb, 0);
    ASSERT_TRUE(results.empty());
    return true;
}

// =============================================================================
// VECTOR STORE TESTS
// =============================================================================

bool test_vector_store_default_constructor() {
    vector_store vs;
    // If constructor completes without throwing, test passes
    return true;
}

bool test_vector_store_custom_constructor() {
    auto collection = make_shared<Collection>();
    auto index = make_shared<FlatIndex>(collection, createSimilarity("cosine"));
    
    vector_store vs(index, collection);
    // If constructor completes without throwing, test passes
    return true;
}

bool test_vector_store_insert() {
    vector_store vs;
    
    vector<float> emb = {1.0f, 2.0f, 3.0f};
    Document doc(emb);
    
    size_t id = vs.insert(doc);
    ASSERT_EQ(0, id);
    return true;
}

bool test_vector_store_multiple_insert() {
    vector_store vs;
    
    vector<Document> docs;
    for (int i = 0; i < 3; i++) {
        vector<float> emb = {static_cast<float>(i), static_cast<float>(i+1)};
        Document doc(emb);
        docs.push_back(doc);
    }
    
    vector<size_t> ids = vs.insert(docs);
    ASSERT_EQ(3, ids.size());
    ASSERT_EQ(0, ids[0]);
    ASSERT_EQ(1, ids[1]);
    ASSERT_EQ(2, ids[2]);
    return true;
}

bool test_vector_store_search() {
    vector_store vs;
    
    // Insert some documents
    vector<float> emb1 = {1.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f};
    Document doc1(emb1);
    Document doc2(emb2);
    
    vs.insert(doc1);
    vs.insert(doc2);
    
    // Search for similar documents
    vector<Document> results = vs.search(emb1, 1);
    ASSERT_EQ(1, results.size());
    ASSERT_TRUE(vectors_equal(emb1, results[0].getEmbedding()));
    return true;
}

bool test_vector_store_search_with_scores() {
    vector_store vs;
    
    vector<float> emb1 = {1.0f, 0.0f};
    vector<float> emb2 = {0.0f, 1.0f};
    Document doc1(emb1);
    Document doc2(emb2);
    
    vs.insert(doc1);
    vs.insert(doc2);
    
    vector<std::pair<float, Document>> results = vs.searchWithScores(emb1, 2);
    ASSERT_EQ(2, results.size());
    ASSERT_NEAR(1.0f, results[0].first, 1e-6f);
    return true;
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

bool test_integration_full_workflow() {
    vector_store vs;
    
    // Create documents with metadata and embeddings
    json meta1 = {{"type", "document"}, {"category", "science"}};
    json meta2 = {{"type", "document"}, {"category", "technology"}};
    json meta3 = {{"type", "document"}, {"category", "science"}};
    
    vector<float> emb1 = {0.8f, 0.6f, 0.1f};
    vector<float> emb2 = {0.2f, 0.9f, 0.4f};
    vector<float> emb3 = {0.7f, 0.5f, 0.2f}; // Similar to emb1
    
    Document doc1(emb1, Metadata(meta1));
    Document doc2(emb2, Metadata(meta2));
    Document doc3(emb3, Metadata(meta3));
    
    // Insert documents
    size_t id1 = vs.insert(doc1);
    size_t id2 = vs.insert(doc2);
    size_t id3 = vs.insert(doc3);
    
    ASSERT_EQ(0, id1);
    ASSERT_EQ(1, id2);
    ASSERT_EQ(2, id3);
    
    // Test similarity search
    vector<Document> similar = vs.search(emb1, 2);
    ASSERT_EQ(2, similar.size());
    
    // Test metadata search
    Metadata scienceMeta(meta1);
    vector<Document> scienceDocs = vs.search(scienceMeta, 5);
    ASSERT_EQ(2, scienceDocs.size());
    
    // Test combined search (metadata + embedding)
    vector<Document> combined = vs.search(scienceMeta, emb1, 1);
    ASSERT_EQ(1, combined.size());
    
    // Test update
    vector<float> newEmb = {0.1f, 0.1f, 0.9f};
    Document newDoc(newEmb, Metadata(meta1));
    vs.update(id1, newDoc);
    
    // Verify update worked
    Document fetchedDoc = vs.fetchDocument(id1);
    ASSERT_TRUE(vectors_equal(newEmb, fetchedDoc.getEmbedding()));
    
    return true;
}

bool test_integration_large_dataset() {
    vector_store vs;
    
    // Insert 100 documents
    vector<Document> docs;
    for (int i = 0; i < 100; i++) {
        vector<float> emb = {
            static_cast<float>(i % 10),
            static_cast<float>((i + 1) % 10),
            static_cast<float>((i + 2) % 10)
        };
        json meta = {{"id", i}, {"batch", i / 10}};
        Document doc(emb, Metadata(meta));
        docs.push_back(doc);
    }
    
    vector<size_t> ids = vs.insert(docs);
    ASSERT_EQ(100, ids.size());
    
    // Test search performance with various k values
    vector<float> queryEmb = {5.0f, 5.0f, 5.0f};
    
    vector<Document> results5 = vs.search(queryEmb, 5);
    ASSERT_EQ(5, results5.size());
    
    vector<Document> results20 = vs.search(queryEmb, 20);
    ASSERT_EQ(20, results20.size());
    
    vector<Document> results100 = vs.search(queryEmb, 100);
    ASSERT_EQ(100, results100.size());
    
    return true;
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

int main() {
    cout << "===========================================" << endl;
    cout << "       StoreX Unit Tests - Phase 1 & 2   " << endl;
    cout << "===========================================" << endl;
    
    // Metadata Tests
    cout << "\n=== METADATA TESTS ===" << endl;
    RUN_TEST(test_metadata_default_constructor);
    RUN_TEST(test_metadata_json_constructor);
    RUN_TEST(test_metadata_string_constructor);
    RUN_TEST(test_metadata_invalid_json_string);
    RUN_TEST(test_metadata_empty_string);
    RUN_TEST(test_metadata_equality);
    RUN_TEST(test_metadata_to_string);
    
    // Document Tests
    cout << "\n=== DOCUMENT TESTS ===" << endl;
    RUN_TEST(test_document_default_constructor);
    RUN_TEST(test_document_embedding_constructor);
    RUN_TEST(test_document_embedding_metadata_constructor);
    RUN_TEST(test_document_setters);
    RUN_TEST(test_document_equality);
    
    // Similarity Tests
    cout << "\n=== SIMILARITY TESTS ===" << endl;
    RUN_TEST(test_cosine_similarity);
    RUN_TEST(test_euclidean_similarity);
    RUN_TEST(test_jaccard_similarity);
    RUN_TEST(test_similarity_factory);
    RUN_TEST(test_similarity_edge_cases);
    
    // Collection Tests
    cout << "\n=== COLLECTION TESTS ===" << endl;
    RUN_TEST(test_collection_default_constructor);
    RUN_TEST(test_collection_insert);
    RUN_TEST(test_collection_update);
    RUN_TEST(test_collection_update_out_of_range);
    RUN_TEST(test_collection_multiple_inserts);
    
    // Storage Tests
    cout << "\n=== STORAGE TESTS ===" << endl;
    RUN_TEST(test_storage_constructor);
    RUN_TEST(test_storage_insert_and_load);
    
    // Flat Index Tests
    cout << "\n=== FLAT INDEX TESTS ===" << endl;
    RUN_TEST(test_flat_index_constructor);
    RUN_TEST(test_flat_index_insert);
    RUN_TEST(test_flat_index_update);
    RUN_TEST(test_flat_index_search_by_metadata);
    RUN_TEST(test_flat_index_search_by_embedding);
    RUN_TEST(test_flat_index_search_with_scores);
    RUN_TEST(test_flat_index_empty_collection_search);
    RUN_TEST(test_flat_index_search_k_zero);
    
    // Vector Store Tests
    cout << "\n=== VECTOR STORE TESTS ===" << endl;
    RUN_TEST(test_vector_store_default_constructor);
    RUN_TEST(test_vector_store_custom_constructor);
    RUN_TEST(test_vector_store_insert);
    RUN_TEST(test_vector_store_multiple_insert);
    RUN_TEST(test_vector_store_search);
    RUN_TEST(test_vector_store_search_with_scores);
    
    // Integration Tests
    cout << "\n=== INTEGRATION TESTS ===" << endl;
    RUN_TEST(test_integration_full_workflow);
    RUN_TEST(test_integration_large_dataset);
    
    // Summary
    cout << "\n===========================================" << endl;
    cout << "              TEST SUMMARY                " << endl;
    cout << "===========================================" << endl;
    cout << "Tests Passed: " << tests_passed << endl;
    cout << "Tests Failed: " << tests_failed << endl;
    cout << "Total Tests:  " << (tests_passed + tests_failed) << endl;
    
    if (tests_failed == 0) {
        cout << "\n🎉 ALL TESTS PASSED! 🎉" << endl;
        cout << "Your StoreX implementation for Phase 1 & 2 is working correctly!" << endl;
        return 0;
    } else {
        cout << "\n❌ SOME TESTS FAILED ❌" << endl;
        cout << "Please review the failed tests and fix the issues." << endl;
        return 1;
    }
}