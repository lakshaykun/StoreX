#include "index.hpp"

// LSHIndex implementation
LSHIndex::LSHIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
    : Index(coll, sim) {
    cout << ">> LSH index created!!\n";
    lsh = makeLSH(coll, sim);
}

size_t LSHIndex::insert(Document& doc) {
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    int id = collection->insert(doc);
    lsh->insert(doc.getEmbedding(), id);
    return id;
}

void LSHIndex::update(size_t id, Document& doc) {
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    Document existingDoc = collection->getDocument(id);
    collection->update(id, doc);
    lsh->update(id, doc.getEmbedding(), existingDoc.getEmbedding());
}

vector<Document> LSHIndex::search(const Metadata& meta, size_t k) const {
    vector<Document> results;
    for (const auto& doc : collection->getDocuments()) {
        if (doc.getMetadata() == meta) {
            results.emplace_back(doc);
            if (results.size() >= k) {
                break;
            }
        }
    }
    return results;
}

vector<Document> LSHIndex::search(const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {};
    }
    return lsh->search(embedding, k);
}

vector<pair<float, Document>> LSHIndex::searchWithScores(const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {};
    }

    return lsh->searchWithScores(embedding, k);
}

vector<Document> LSHIndex::search(const Metadata& meta, const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {};
    }
    return lsh->search(meta, embedding, k);
}

vector<pair<float, Document>> LSHIndex::searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {};
    }

    return lsh->searchWithScores(meta, embedding, k);
}