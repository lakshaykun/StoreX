#include "lsh.hpp"
#include <cstddef>
#include <memory>

// <<<<< For Jaccard >>>>>
vector<int> LSHJaccard::binarize(const vector<float>& embedding) const {
    vector<int> binary(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        binary[i] = embedding[i] > 0 ? 1 : 0;
    }
    return binary;
}

LSHJaccard::LSHJaccard(size_t nHashFunctions, size_t nBands, size_t rowsPerBand, std::shared_ptr<Collection> coll, std::shared_ptr<Similarity> sim)
    : numHashFunctions(nHashFunctions), numBands(nBands), rowsPerBand(rowsPerBand), vecSize(coll->getVectorSize()), LSH(coll, sim) {

    // Validate parameters
    if (numBands * rowsPerBand != numHashFunctions) {
        throw std::invalid_argument("numBands * rowsPerBand must equal numHashFunctions");
    }
    
    bands.resize(numBands);
    hashFunctions.resize(numHashFunctions);
    
    // Initialize hash functions with random permutations
    std::random_device rd;
    std::mt19937 g(rd());
    
    for (size_t i = 0; i < numHashFunctions; ++i) {
        hashFunctions[i].resize(vecSize);
        std::iota(hashFunctions[i].begin(), hashFunctions[i].end(), 0); // Initialize with sequential numbers
        std::shuffle(hashFunctions[i].begin(), hashFunctions[i].end(), g);
    }
}

// Method to compute signature for a given embedding
vector<int> LSHJaccard::computeSignature(const vector<int>& embedding) const {
    vector<int> signature(numHashFunctions, std::numeric_limits<int>::max());
    for (size_t i = 0; i < numHashFunctions; ++i) {
        for (size_t j = 0; j < vecSize; ++j) {
            if (embedding[hashFunctions[i][j]] != 0) { // Only consider non-zero elements
                signature[i] = hashFunctions[i][j];
                break;
            }
        }
    }
    return signature;
}

// Method to hash a document's embedding into bands
void LSHJaccard::insert(const vector<float>& embedding, std::size_t docId) {
    vector<int> signature = computeSignature(binarize(embedding));
    for (size_t band = 0; band < numBands; ++band) {
        vector<int> bandSignature(signature.begin() + band * rowsPerBand, 
                                    signature.begin() + (band + 1) * rowsPerBand);
        bands[band][bandSignature].insert(docId);
    }
}

// Method to update a document's embedding in the LSH
void LSHJaccard::update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) {
    // Remove the old embedding
    vector<int> oldSignature = computeSignature(binarize(oldEmbedding));
    for (size_t band = 0; band < numBands; ++band) {
        vector<int> bandSignature(oldSignature.begin() + band * rowsPerBand, 
                                    oldSignature.begin() + (band + 1) * rowsPerBand);
        bands[band][bandSignature].erase(docId);
    }
    
    // Insert the new embedding
    insert(newEmbedding, docId);
}

// Method to search for similar documents based on a query embedding
vector<Document> LSHJaccard::search(const vector<float>& query, size_t k) const {
    vector<int> queryEmbedding = binarize(query);
    vector<int> querySignature = computeSignature(queryEmbedding);
    std::unordered_set<size_t> resultSet;
    
    for (size_t band = 0; band < numBands; ++band) {
        vector<int> bandSignature(querySignature.begin() + band * rowsPerBand, 
                                    querySignature.begin() + (band + 1) * rowsPerBand);
        auto it = bands[band].find(bandSignature);
        if (it != bands[band].end()) {
            const auto& docIds = it->second;
            resultSet.insert(docIds.begin(), docIds.end());
        }
    }
    
    // Convert to vector and limit to k results
    vector<size_t> candidates(resultSet.begin(), resultSet.end());
    vector<pair<float, size_t>> scoredDocuments;
    scoredDocuments.reserve(candidates.size());

    for (const auto& docId : candidates) {
        float score = similarity->compute(query, collection->getDocument(docId).getEmbedding());
        scoredDocuments.emplace_back(score, docId);
    }
    k = std::min(k, scoredDocuments.size());
    std::partial_sort(scoredDocuments.begin(), scoredDocuments.begin() + k, scoredDocuments.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract the top k document IDs
    vector<Document> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        results.push_back(collection->getDocument(scoredDocuments[i].second));
    }
    return results;
}


// Method to search for similar documents and their scores by embedding
vector<pair<float, Document>> LSHJaccard::searchWithScores(const vector<float>& query, size_t k) const {
    vector<int> queryEmbedding = binarize(query);
    vector<int> querySignature = computeSignature(queryEmbedding);
    std::unordered_set<size_t> resultSet;
    
    for (size_t band = 0; band < numBands; ++band) {
        vector<int> bandSignature(querySignature.begin() + band * rowsPerBand, 
                                    querySignature.begin() + (band + 1) * rowsPerBand);
        auto it = bands[band].find(bandSignature);
        if (it != bands[band].end()) {
            const auto& docIds = it->second;
            resultSet.insert(docIds.begin(), docIds.end());
        }
    }
    
    // Convert to vector and limit to k results
    vector<size_t> candidates(resultSet.begin(), resultSet.end());
    vector<pair<float, size_t>> scoredDocuments;
    scoredDocuments.reserve(candidates.size());

    for (const auto& docId : candidates) {
        float score = similarity->compute(query, collection->getDocument(docId).getEmbedding());
        scoredDocuments.emplace_back(score, docId);
    }
    k = std::min(k, scoredDocuments.size());
    std::partial_sort(scoredDocuments.begin(), scoredDocuments.begin() + k, scoredDocuments.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract the top k document IDs and their scores
    vector<pair<float, Document>> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        Document doc = collection->getDocument(scoredDocuments[i].second);
        results.emplace_back(scoredDocuments[i].first, doc);
    }
    return results;
}


// Method to search for top k similar documents by embedding with same metadata
vector<Document> LSHJaccard::search(const Metadata& meta, const vector<float>& query, size_t k) const {
    vector<int> queryEmbedding = binarize(query);
    vector<int> querySignature = computeSignature(queryEmbedding);
    std::unordered_set<size_t> resultSet;
    
    for (size_t band = 0; band < numBands; ++band) {
        vector<int> bandSignature(querySignature.begin() + band * rowsPerBand, 
                                    querySignature.begin() + (band + 1) * rowsPerBand);
        auto it = bands[band].find(bandSignature);
        if (it != bands[band].end()) {
            const auto& docIds = it->second;
            resultSet.insert(docIds.begin(), docIds.end());
        }
    }
    
    // Convert to vector and limit to k results
    vector<size_t> candidates(resultSet.begin(), resultSet.end());
    vector<pair<float, size_t>> scoredDocuments;

    for (const auto& docId : candidates) {
        if (collection->getDocument(docId).getMetadata() == meta) {
            float score = similarity->compute(query, collection->getDocument(docId).getEmbedding());
            scoredDocuments.emplace_back(score, docId);
        }
    }
    k = std::min(k, scoredDocuments.size());
    std::partial_sort(scoredDocuments.begin(), scoredDocuments.begin() + k, scoredDocuments.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract the top k document IDs
    vector<Document> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        results.push_back(collection->getDocument(scoredDocuments[i].second));
    }
    return results;
}

// Method to search for similar documents and their scores by embedding with same metadata
vector<pair<float, Document>> LSHJaccard::searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const {
    vector<int> queryEmbedding = binarize(query);
    vector<int> querySignature = computeSignature(queryEmbedding);
    std::unordered_set<size_t> resultSet;
    
    for (size_t band = 0; band < numBands; ++band) {
        vector<int> bandSignature(querySignature.begin() + band * rowsPerBand, 
                                    querySignature.begin() + (band + 1) * rowsPerBand);
        auto it = bands[band].find(bandSignature);
        if (it != bands[band].end()) {
            const auto& docIds = it->second;
            resultSet.insert(docIds.begin(), docIds.end());
        }
    }
    
    // Convert to vector and limit to k results
    vector<size_t> candidates(resultSet.begin(), resultSet.end());
    vector<pair<float, size_t>> scoredDocuments;

    for (const auto& docId : candidates) {
        if (collection->getDocument(docId).getMetadata() == meta) {
            float score = similarity->compute(query, collection->getDocument(docId).getEmbedding());
            scoredDocuments.emplace_back(score, docId);
        }
    }
    k = std::min(k, scoredDocuments.size());
    std::partial_sort(scoredDocuments.begin(), scoredDocuments.begin() + k, scoredDocuments.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract the top k document IDs and their scores
    vector<pair<float, Document>> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        Document doc = collection->getDocument(scoredDocuments[i].second);
        results.emplace_back(scoredDocuments[i].first, doc);
    }
    return results;
}















// <<<<< For Cosine >>>>>
LSHCosine::LSHCosine(size_t nBits, size_t nBands, shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
    : numBits(nBits), numBands(nBands), vecSize(coll->getVectorSize()), LSH(coll, sim) {
    bands.resize(numBands);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0, 1.0);
    // Initialize hash functions
    hashFunctions.resize(numBands);
    for (size_t i = 0; i < numBands; ++i) {
        hashFunctions[i].resize(numBits);
        for (size_t j = 0; j < numBits; ++j) {
            hashFunctions[i][j].resize(vecSize);
            for (size_t k = 0; k < vecSize; ++k) {
                hashFunctions[i][j][k] = dist(gen); // Random normal distribution
            }
        }
    }
}

// Method to compute signature for a given embedding
vector<vector<int>> LSHCosine::computeSignature(const vector<float>& embedding) const {
    vector<vector<int>> signature(numBands, vector<int>(numBits));
    for (size_t i = 0; i < numBands; ++i) {
        for (size_t j = 0; j < numBits; ++j) {
            signature[i][j] = utility::dotProduct(embedding, hashFunctions[i][j]);
        }
    }
    return signature;
}

// Method to hash a document's embedding into bands
void LSHCosine::insert(const vector<float>& embedding, size_t docId) {
    vector<vector<int>> signature = computeSignature(embedding);
    for (size_t band=0; band<numBands; band++){
        bands[band][signature[band]].insert(docId);
    }
}

// Method to update a document's embedding in the LSH
void LSHCosine::update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) {
    vector<vector<int>> signature = computeSignature(oldEmbedding);
    for (size_t band=0; band<numBands; band++){
        bands[band][signature[band]].erase(docId);
    }
    insert(newEmbedding, docId);
}

// Method to search for similar documents based on a query embedding
vector<Document> LSHCosine::search(const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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
vector<pair<float, Document>> LSHCosine::searchWithScores(const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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
vector<Document> LSHCosine::search(const Metadata& meta, const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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
vector<pair<float, Document>> LSHCosine::searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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








// <<<<< For Euclidean >>>>>
LSHEuclidean::LSHEuclidean(size_t nBits, size_t nBands, size_t bWidth, shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
    : numBits(nBits), numBands(nBands), bucketWidth(bWidth), vecSize(coll->getVectorSize()), LSH(coll, sim) {
    bands.resize(numBands);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0, 1.0);
    std::uniform_real_distribution<float> distB(0.0f, bucketWidth);
    
    // Initialize hash functions
    hashFunctions.resize(numBands);
    offsets.resize(numBands);
    for (size_t i = 0; i < numBands; ++i) {
        hashFunctions[i].resize(numBits);
        offsets[i].resize(numBits);
        for (size_t j = 0; j < numBits; ++j) {
            offsets[i][j] = distB(gen);
            hashFunctions[i][j].resize(vecSize);
            for (size_t k = 0; k < vecSize; ++k) {
                hashFunctions[i][j][k] = dist(gen); // Random normal distribution
            }
        }
    }
}


// Method to compute signature for a given embedding
vector<vector<int>> LSHEuclidean::computeSignature(const vector<float>& embedding) const {
    vector<vector<int>> signature(numBands, vector<int>(numBits));
    for (size_t i = 0; i < numBands; ++i) {
        for (size_t j = 0; j < numBits; ++j) {
            signature[i][j] = (utility::dotProduct(embedding, hashFunctions[i][j]) + offsets[i][j]) / bucketWidth;
        }
    }
    return signature;
}

// Method to hash a document's embedding into bands
void LSHEuclidean::insert(const vector<float>& embedding, size_t docId) {
    vector<vector<int>> signature = computeSignature(embedding);
    for (size_t band=0; band<numBands; band++){
        bands[band][signature[band]].insert(docId);
    }
}

// Method to update a document's embedding in the LSH
void LSHEuclidean::update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) {
    vector<vector<int>> signature = computeSignature(oldEmbedding);
    for (size_t band=0; band<numBands; band++){
        bands[band][signature[band]].erase(docId);
    }
    insert(newEmbedding, docId);
}

// Method to search for similar documents based on a query embedding
vector<Document> LSHEuclidean::search(const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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
vector<pair<float, Document>> LSHEuclidean::searchWithScores(const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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
vector<Document> LSHEuclidean::search(const Metadata& meta, const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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
vector<pair<float, Document>> LSHEuclidean::searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const {
    std::unordered_set<size_t> candidateSet;
    vector<vector<int>> querySign = computeSignature(query);
    for (size_t band = 0; band < numBands; band++){
        auto it = bands[band].find(querySign[band]);
        if (it != bands[band].end()){
            candidateSet.insert(it->second.begin(), it->second.end());
        }
    }
    vector<size_t> candidates(candidateSet.begin(), candidateSet.end());
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












// Utility function to make lsh
shared_ptr<LSH> makeLSH(shared_ptr<Collection>& coll, shared_ptr<Similarity>& sim){
    if (sim->name == "JaccardSimilarity") {
        return make_shared<LSHJaccard>(10, 5, 2, coll, sim);
    } else if (sim->name == "CosineSimilarity") {
        return make_shared<LSHCosine>(10, 5, coll, sim);
    } else if (sim->name == "EuclideanSimilarity") {
        return make_shared<LSHEuclidean>(10, 5, 2, coll, sim);
    } 
    throw std::invalid_argument("This similarity metric:" + sim->name + " is not supported by LSHIndex!!!");
    return nullptr;
}