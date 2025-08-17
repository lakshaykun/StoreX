#pragma once
#include <string>
#include <vector>
#include <memory>
#include "collection.hpp"
#include "similarity.hpp"
#include "lsh.hpp"

using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::vector;

static thread_local std::mt19937 rng{std::random_device{}()};


// Abstract virtual class to create indexing for vector store
class Index {
protected:
    shared_ptr<Collection> collection;
    shared_ptr<Similarity> similarity;
public:
    virtual ~Index() = default;

    // Constructor to initialize the index with a collection and similarity measure
    Index(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
        : collection(coll), similarity(sim) {}

    // Method to insert a document into the index
    virtual size_t insert(Document& doc) = 0;

    // Method to update a document in the index
    virtual void update(size_t id, Document& doc) = 0;

    // Method to search for documents of a specific metadata
    virtual vector<Document> search(const Metadata& meta, size_t k) const = 0;

    // Method to search for top k similar documents by embedding
    virtual vector<Document> search(const vector<float>& embedding, size_t k) const = 0;

    // Method to search for similar documents and their scores by embedding
    virtual vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k) const = 0;

    // Method to search for top k similar documents by embedding with same metadata
    virtual vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k) const = 0;

    // Method to search for similar documents and their scores by embedding with same metadata
    virtual vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const = 0;
};

class FlatIndex : public Index {
public:
    FlatIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    size_t insert(Document& doc) override;

    void update(size_t id, Document& doc) override;

    vector<Document> search(const Metadata& meta, size_t k) const override;

    vector<Document> search(const vector<float>& embedding, size_t k) const override;

    vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k) const override;

    vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k) const override;

    vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const override;

    ~FlatIndex() override = default;
};

class LSHIndex : public Index {
private:
    shared_ptr<LSH> lsh; // Pointer to the LSH object for indexing
public:
    LSHIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    size_t insert(Document& doc) override;

    void update(size_t id, Document& doc) override;

    vector<Document> search(const Metadata& meta, size_t k) const override;

    vector<Document> search(const vector<float>& embedding, size_t k) const override;

    vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k) const override;
    
    vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k) const override;
    
    vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const override;

    ~LSHIndex() override = default;
};


class AnnoyIndex : public Index {
private:
    size_t numTrees;
    size_t leafSize;    
    // structure of a Node in annoy tree
    struct Node {
        bool isLeaf;
        vector<size_t> docIds;   // only if leaf
        vector<float> normal;   // hyperplane normal
        float offset;                // hyperplane offset
        unique_ptr<Node> left;
        unique_ptr<Node> right;
    };

    // forest of Annoy trees
    vector<unique_ptr<Node>> forest;

    // Method to build the annoy tree
    unique_ptr<Node> buildTree(const vector<size_t>& docIds);

    // Method to build the forest of numTrees annoy tree
    void buildForest();

    // Method to search in the annoy tree
    void searchTree(const unique_ptr<Node>& node, const vector<float>& query, std::unordered_set<size_t>& candidates) const;
    
public:
    AnnoyIndex(size_t num_trees, size_t leaf_size, shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    size_t insert(Document& doc) override;

    void update(size_t id, Document& doc) override;

    vector<Document> search(const Metadata& meta, size_t k) const override;

    vector<Document> search(const vector<float>& embedding, size_t k) const override;

    vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k) const override;
    
    vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k) const override;
    
    vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const override;

    ~AnnoyIndex() override = default;
};