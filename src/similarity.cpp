#include "similarity.hpp"
#include <cmath>
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>

using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;

// Forward declaration of the Similarity class
class Similarity;

// cosine similarity
class CosineSimilarity : public Similarity {
public:
    // Method to compute the cosine similarity score between two vectors
    float compute(const vector<float>& emb1, const vector<float>& emb2) const override {
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
};

// Euclidean distance
class EuclideanSimilarity : public Similarity {
public:
    // Method to compute the Euclidean distance between two vectors
    float compute(const vector<float>& emb1, const vector<float>& emb2) const override {
        if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
            throw invalid_argument("Embeddings must be non-empty and of the same size.");
        }
        
        float sum = 0.0f;
        for (size_t i = 0; i < emb1.size(); ++i) {
            float diff = emb1[i] - emb2[i];
            sum += diff * diff;
        }
        
        return sqrt(sum);
    }
};

// Dot product similarity
class DotProductSimilarity : public Similarity {
public:
    // Method to compute the dot product similarity score between two vectors
    float compute(const vector<float>& emb1, const vector<float>& emb2) const override {
        if (emb1.empty() || emb2.empty() || emb1.size() != emb2.size()) {
            throw invalid_argument("Embeddings must be non-empty and of the same size.");
        }
        
        float dotProduct = 0.0f;
        for (size_t i = 0; i < emb1.size(); ++i) {
            dotProduct += emb1[i] * emb2[i];
        }
        
        return dotProduct;
    }
};

// Factory function to create a similarity instance based on type
shared_ptr<Similarity> createSimilarity(const string& type) {
    if (type == "cosine") {
        return make_shared<CosineSimilarity>();
    } else if (type == "euclidean") {
        return make_shared<EuclideanSimilarity>();
    } else if (type == "dot_product") {
        return make_shared<DotProductSimilarity>();
    } else {
        throw invalid_argument("Unknown similarity type: " + type);
    }
}