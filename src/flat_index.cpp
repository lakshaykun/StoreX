#include "index.hpp"
#include "collection.hpp"
#include "similarity.hpp"
#include "document.hpp"
#include <memory>
#include <vector>
#include <algorithm>

using std::move;
using std::shared_ptr;
using std::vector;
using std::sort;

class FlatIndex : public Index {
protected:
    shared_ptr<Collection> collection;
    shared_ptr<Similarity> similarity;
public:
    // Constructor to initialize the FlatIndex with a collection and similarity measure
    FlatIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
        : Index(move(coll), move(sim)) {}

    // Destructor
    ~FlatIndex() override = default;

    // Method to insert a document into the index
    int insert(Document& doc) override {
        return collection->insert(doc);
    }
    
    // Method to search for documents of a specific metadata
    vector<Document> search(const Metadata& meta, int k) const override {
        vector<Document> results;
        // Iterate through the collection and find documents matching the metadata
        for (const auto& doc : collection->documents) {
            if (doc.getMetadata() == meta) {
                results.push_back(doc);
                if (results.size() >= k) {
                    break; // Stop if we have enough results
                }
            }
        }
        return results;
    }
    
    // Method to search for similar documents by embedding
    vector<Document> search(const vector<float>& embedding, int k) const override {
        vector<Document> results;
        // Iterate through the collection and compute similarity scores
        for (const auto& doc : collection->documents) {
            float score = similarity->compute(doc.getEmbedding(), embedding);
            if (results.size() < k || score > results.back().getEmbedding().size()) {
                results.push_back(doc);
                // Sort results based on similarity score
                sort(results.begin(), results.end(), 
                          [this, &embedding](const Document& a, const Document& b) {
                              return similarity->compute(a.getEmbedding(), embedding) > 
                                     similarity->compute(b.getEmbedding(), embedding);
                          });
                if (results.size() > k) {
                    results.pop_back();
                }
            }
        }
        return results;
    }

    // Method to search for top k similar documents by embedding with same metadata
    vector<Document> search(const Metadata& meta, const vector<float>& embedding, int k) const override {
        vector<Document> results;
        // Iterate through the collection and find documents matching the metadata
        for (const auto& doc : collection->documents) {
            if (doc.getMetadata() == meta) {
                float score = similarity->compute(doc.getEmbedding(), embedding);
                if (results.size() < k || score > results.back().getEmbedding().size()) {
                    results.push_back(doc);
                    // Sort results based on similarity score
                    sort(results.begin(), results.end(), 
                              [this, &embedding](const Document& a, const Document& b) {
                                  return similarity->compute(a.getEmbedding(), embedding) > 
                                         similarity->compute(b.getEmbedding(), embedding);
                              });
                    if (results.size() > k) {
                        results.pop_back();
                    }
                }
            }
        }
        return results;
    }
};
