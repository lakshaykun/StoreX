#include "index.hpp"
#include "utility.hpp"
#include <unordered_set>
#include <random>
#include <algorithm>
#include <numeric>

AnnoyIndex::AnnoyIndex(size_t num_trees, size_t leaf_size, shared_ptr<Collection> coll, shared_ptr<Similarity> sim) 
    : Index(coll, sim), numTrees(num_trees), leafSize(leaf_size)
{   
    if (!coll || !sim) {
        throw std::invalid_argument("Collection and similarity cannot be null");
    }
    if (num_trees == 0 || leaf_size == 0) {
        throw std::invalid_argument("num_trees and leaf_size must be positive");
    }
    rng.seed(std::random_device{}());
    forest.resize(numTrees);
    buildForest();
}

// Method to build the annoy tree
unique_ptr<AnnoyIndex::Node> AnnoyIndex::buildTree(const vector<size_t>& docIds){
    if (docIds.size() <= leafSize) {
        unique_ptr<Node> leaf = std::make_unique<Node>();
        leaf->isLeaf = true;
        leaf->docIds = docIds;
        return leaf;
    }

    // random points from docIds
    std::uniform_int_distribution<size_t> dist(0, docIds.size() - 1);
    size_t i = dist(rng);
    size_t j = dist(rng);
    while (i == j) {
        j = dist(rng);
    }

    // get the random embeddings
    vector<float> v1 = collection->getDocument(docIds[i]).getEmbedding();
    vector<float> v2 = collection->getDocument(docIds[j]).getEmbedding();

    // compute normal hyperplane (v1 - v2)
    vector<float> normal(v1.size());
    for (size_t k = 0; k < v1.size(); k++) normal[k] = v1[k] - v2[k];

    // compute offset = midpoint · normal
    float offset = 0;
    for (size_t k = 0; k < v1.size(); k++) offset += (v1[k] + v2[k]) * normal[k] / 2;

    // partition ids in left and right
    vector<size_t> leftIds, rightIds;
    for (size_t id : docIds) {
        float dot = utility::dotProduct(collection->getDocument(id).getEmbedding(), normal);
        if (dot > offset) {
            rightIds.push_back(id);
        } else {
            leftIds.push_back(id);
        }
    }

    // Handle edge case where all points go to one side
    if (leftIds.empty() || rightIds.empty()) {
        // Fallback to creating a leaf node
        unique_ptr<Node> leaf = std::make_unique<Node>();
        leaf->isLeaf = true;
        leaf->docIds = docIds;
        return leaf;
    }

    // create Node for current
    unique_ptr<Node> node = std::make_unique<Node>();
    node->isLeaf = false;
    node->normal = normal;
    node->offset = offset;
    node->left = move(buildTree(leftIds));
    node->right = move(buildTree(rightIds));

    return node;
}

// Method to build the forest of numTrees annoy tree
void AnnoyIndex::buildForest(){
    vector<size_t> ids(collection->size());
    std::iota(ids.begin(), ids.end(), 0);
    for (auto& tree: forest){
        tree = move(buildTree(ids));
    }
}

// Method to search in the annoy tree
void AnnoyIndex::searchTree(const unique_ptr<Node>& node, const vector<float>& query, std::unordered_set<size_t>& candidates) const {
    if (node->isLeaf) {
        for (const auto& id : node->docIds) {
            candidates.insert(id);
        }
        return;
    }

    float dot = utility::dotProduct(node->normal, query);
    if (dot > node->offset) {
        searchTree(node->right, query, candidates);
    } else {
        searchTree(node->left, query, candidates);
    }
}

// Method to insert a document into the index
size_t AnnoyIndex::insert(Document& doc) {
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    size_t docId = collection->insert(doc);
    buildForest();
    return docId;
}

// Method to update a document in the index
void AnnoyIndex::update(size_t id, Document& doc){
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    collection->update(id, doc);
    buildForest();
}

// Method to search for documents of a specific metadata
vector<Document> AnnoyIndex::search(const Metadata& meta, size_t k) const {
    vector<Document> results;
    // Iterate through the collection and find documents matching the metadata
    for (const auto& doc : collection->getDocuments()) {
        if (doc.getMetadata() == meta) {
            results.emplace_back(doc);
            if (results.size() >= k) {
                break; // Stop if we have enough results
            }
        }
    }
    return results;
}

// Method to search for similar documents based on a query embedding
vector<Document> AnnoyIndex::search(const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidatesSet;
    for (auto& root: forest){
        searchTree(root, query, candidatesSet);
    }
    vector<size_t> candidates(candidatesSet.begin(), candidatesSet.end());
    vector<pair<float, size_t>> scoredDocs;
    for (auto& docId: candidates){
        scoredDocs.emplace_back(similarity->compute(query, collection->getDocument(docId).getEmbedding()), docId);
    }
    
    // sorting the top k docs
    k = std::min(k, scoredDocs.size());
    std::partial_sort(scoredDocs.begin(), scoredDocs.begin() + k, scoredDocs.end(), 
                    [](const auto& a, const auto& b) { return a.first > b.first; });
    
    vector<Document> res(k);
    for (size_t i=0; i<k; i++){
        res[i] = collection->getDocument(scoredDocs[i].second);
    }
    return res;
}


// Method to search for similar documents and their scores by embedding
vector<pair<float, Document>> AnnoyIndex::searchWithScores(const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidatesSet;
    for (auto& root: forest){
        searchTree(root, query, candidatesSet);
    }
    vector<size_t> candidates(candidatesSet.begin(), candidatesSet.end());
    vector<pair<float, size_t>> scoredDocs;
    for (auto& docId: candidates){
        scoredDocs.emplace_back(similarity->compute(query, collection->getDocument(docId).getEmbedding()), docId);
    }
    
    // sorting the top k docs
    k = std::min(k, scoredDocs.size());
    std::partial_sort(scoredDocs.begin(), scoredDocs.begin() + k, scoredDocs.end(), 
                    [](const auto& a, const auto& b) { return a.first > b.first; });
    
    vector<pair<float, Document>> res(k);
    for (size_t i=0; i<k; i++){
        res[i].first = scoredDocs[i].first;
        res[i].second = collection->getDocument(scoredDocs[i].second);
    }
    return res;
}

// Method to search for top k similar documents by embedding with same metadata
vector<Document> AnnoyIndex::search(const Metadata& meta, const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidatesSet;
    for (auto& root: forest){
        searchTree(root, query, candidatesSet);
    }
    vector<size_t> candidates(candidatesSet.begin(), candidatesSet.end());
    vector<pair<float, size_t>> scoredDocs;
    for (auto& docId: candidates){
        if (collection->getDocument(docId).getMetadata() == meta){
            scoredDocs.emplace_back(similarity->compute(query, collection->getDocument(docId).getEmbedding()), docId);
        }
    }
    
    // sorting the top k docs
    k = std::min(k, scoredDocs.size());
    std::partial_sort(scoredDocs.begin(), scoredDocs.begin() + k, scoredDocs.end(), 
                    [](const auto& a, const auto& b) { return a.first > b.first; });
    
    vector<Document> res(k);
    for (size_t i=0; i<k; i++){
        res[i] = collection->getDocument(scoredDocs[i].second);
    }
    return res;
}

// Method to search for similar documents and their scores by embedding with same metadata
vector<pair<float, Document>> AnnoyIndex::searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidatesSet;
    for (auto& root: forest){
        searchTree(root, query, candidatesSet);
    }
    vector<size_t> candidates(candidatesSet.begin(), candidatesSet.end());
    vector<pair<float, size_t>> scoredDocs;
    for (auto& docId: candidates){
        if (collection->getDocument(docId).getMetadata() == meta){
            scoredDocs.emplace_back(similarity->compute(query, collection->getDocument(docId).getEmbedding()), docId);
        }
    }
    
    // sorting the top k docs
    k = std::min(k, scoredDocs.size());
    std::partial_sort(scoredDocs.begin(), scoredDocs.begin() + k, scoredDocs.end(), 
                    [](const auto& a, const auto& b) { return a.first > b.first; });
    
    vector<pair<float, Document>> res(k);
    for (size_t i=0; i<k; i++){
        res[i].first = scoredDocs[i].first;
        res[i].second = collection->getDocument(scoredDocs[i].second);
    }
    return res;
}