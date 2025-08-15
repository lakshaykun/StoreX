#include "similarity.hpp"

// Method to compute the cosine similarity score between two vectors
float CosineSimilarity::compute(const vector<float>& emb1, const vector<float>& emb2) const {
    if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
        throw invalid_argument("Embeddings must be non-empty and of the same size.");
    }
    
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    for (size_t i = 0; i < emb1.size(); ++i) {
        dotProduct += emb1[i] * emb2[i];
        normA += emb1[i] * emb1[i];
        normB += emb2[i] * emb2[i];
    }
    
    if (normA == 0.0f || normB == 0.0f) {
        throw invalid_argument("One of the embeddings is zero.");
    }
    
    return dotProduct / (sqrt(normA) * sqrt(normB));
}

// Method to compute the Euclidean distance between two vectors
float EuclideanSimilarity::compute(const vector<float>& emb1, const vector<float>& emb2) const {
    if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
        throw invalid_argument("Embeddings must be non-empty and of the same size.");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < emb1.size(); ++i) {
        float diff = emb1[i] - emb2[i];
        sum += diff * diff;
    }
    
    return exp(-sqrt(sum));
}

// Method to compute the dot product similarity between two vectors
float JaccardSimilarity::compute(const vector<float>& emb1, const vector<float>& emb2) const {
    if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
        throw invalid_argument("Embeddings must be non-empty and of the same size.");
    }
    
    float intersection = 0.0f;
    float unionSet = 0.0f;
    
    for (size_t i = 0; i < emb1.size(); ++i) {
        if (emb1[i] > 0 && emb2[i] > 0) {
            intersection += std::min(emb1[i], emb2[i]);
        }
        unionSet += std::max(emb1[i], emb2[i]);
    }
    
    if (unionSet == 0.0f) {
        return 0.0f; // Avoid division by zero
    }
    
    return intersection / unionSet;
}

shared_ptr<Similarity> createSimilarity(const string& type) {
    if (type == "cosine") {
        return make_shared<CosineSimilarity>();
    } else if (type == "euclidean") {
        return make_shared<EuclideanSimilarity>();
    } else if (type == "jaccard") {
        return make_shared<JaccardSimilarity>();
    } else {
        throw invalid_argument("Unknown similarity type: " + type);
    }
}