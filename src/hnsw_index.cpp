#include "index.hpp"
#include <queue>

// Method to generate a random level for a new node
size_t HNSWIndex::randomLevel(){
    float r = std::max((float)rand()/RAND_MAX, 1e-6f);
    return std::min((size_t)(-log(r) * mL), maxPossibleLevel);  // geometric distribution
}

// Method to perform greedy descent search
size_t HNSWIndex::greedyDescent(size_t level, const vector<float>& emb) const {
    size_t currLevel = maxLevel;
    size_t ep = entryPoint;
    auto epEmb = collection->getDocument(ep).getEmbedding();
    for (; currLevel > level; currLevel--){
        bool changed = 1;
        while (changed){
            changed = 0;
            for (auto& neighbour: adj[currLevel].at(ep)){
                auto nEmb = collection->getDocument(neighbour).getEmbedding();
                // if neighbour is more similary go to neighbour
                if (similarity->compute(nEmb, emb) > similarity->compute(epEmb, emb)){
                    ep = neighbour;
                    epEmb = nEmb;
                    changed = true;
                }
            }
        }
    }
    return ep;
}

// Search for candidates in a layer
vector<size_t> HNSWIndex::searchLayer(size_t ep, size_t level, const vector<float>& emb) const {
    auto cmp = [&](const auto& a, const auto& b) {
        return a.first < b.first; // max-heap (worst distance on top)
    };
    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, decltype(cmp)> candidates(cmp);
    unordered_set<size_t> visited;
    // start point
    float d = similarity->compute(emb, collection->getDocument(ep).getEmbedding());
    candidates.push({d, ep});
    visited.insert(ep);
    // add nodes into candidates until no further improvement
    while (!candidates.empty()){
        bool improve = 0;

        auto [worstSim, node] = candidates.top();
        for (auto& neigh: adj[level].at(node)){
            if (visited.count(neigh)) continue;
            visited.insert(neigh);
            d = similarity->compute(collection->getDocument(neigh).getEmbedding(), emb);
            // if current node can be added
            if (candidates.size() < efConstruction || d > candidates.top().first) {
                candidates.push({d, neigh});
                if (candidates.size() > efConstruction) candidates.pop();
                improve = true;
            }
        }

        if (!improve) break;
    }

    // return just node IDs
    vector<size_t> result;
    while (!candidates.empty()) {
        result.push_back(candidates.top().second);
        candidates.pop();
    }
    return result;
}

// Method to select maxDeg neighbors from candidates
vector<size_t> HNSWIndex::selectNeighbors(const vector<size_t>& candidates, size_t maxDeg, const vector<float>& emb) const {
    vector<pair<float, size_t>> dists;
    for (auto& c: candidates){
        dists.emplace_back(similarity->compute(emb, collection->getDocument(c).getEmbedding()), c);
    }
    sort(dists.begin(), dists.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    vector<size_t> results;
    for (auto [d, c]: dists){
        bool ok = 1;
        auto cEmb = collection->getDocument(c).getEmbedding(); 
        for (auto r : results) {
            float dr = similarity->compute(cEmb, collection->getDocument(r).getEmbedding());
            if (dr > d) {
                // too close to a result
                ok = false;
                break;
            }            
        }
        if (ok) results.push_back(c);
        if (results.size() >= maxDeg) break;
    }
    return results;
}

// Method to prune neighbors to maintain maxDeg
void HNSWIndex::pruneNeighbors(size_t layer, size_t node, size_t maxDeg){
    vector<size_t> neighbors(adj[layer][node].begin(), adj[layer][node].end());
    auto selected = selectNeighbors(neighbors, maxDeg,
                                             collection->getDocument(node).getEmbedding());
    adj[layer][node].clear();
    for (auto s : selected) {
        adj[layer][node].insert(s);
    }
}

// Method to find the closest candidate to emb
size_t HNSWIndex::closestCandidate(const vector<size_t>& candidates, const vector<float>& emb) const {
    float bestDist = -1e9; // since we're maximizing similarity
    size_t bestId = candidates[0];
    for (auto c : candidates) {
        float d = similarity->compute(collection->getDocument(c).getEmbedding(), emb);
        if (d > bestDist) {
            bestDist = d;
            bestId = c;
        }
    }
    return bestId;
}

// Method to insert a new node into the HNSW graph
void HNSWIndex::insertHNSW(size_t docId, Document& doc){
    size_t level = randomLevel();
    const auto& docEmb = doc.getEmbedding();
    
    // Handle first insertion (empty graph)
    if (entryPoint == SIZE_MAX) {
        // Initialize the adjacency list layers up to the new node's level
        while (adj.size() <= level) {
            adj.push_back({});
        }
        // Add empty adjacency list for the new node at each level
        for (size_t l = 0; l <= level; l++) {
            adj[l][docId] = {};
        }
        maxLevel = level;
        entryPoint = docId;
        return;
    }
    
    size_t ep = entryPoint, currL = maxLevel;

    if (level > maxLevel) {
        // Extend adjacency list to new level
        while (adj.size() <= level) {
            adj.push_back({});
        }
        // Initialize empty adjacency for docId at all levels
        for (size_t l = maxLevel + 1; l <= level; l++) {
            adj[l][docId] = {};
        }
        maxLevel = level;
        entryPoint = docId;
        currL = level;
    } else {
        ep = greedyDescent(level, doc.getEmbedding());
        currL = level;
    }

    auto epEmb = collection->getDocument(ep).getEmbedding();
    for (int currLevel=currL; currLevel>=0; currLevel--){
        // Ensure adjacency list exists for this node at this level
        if (adj[currLevel].find(docId) == adj[currLevel].end()) {
            adj[currLevel][docId] = {};
        }
        
        auto candidates = searchLayer(ep, currLevel, docEmb);
        auto selected = selectNeighbors(candidates, (currLevel == 0 ? Mmax0 : M), docEmb);

        // add selected connections
        for (auto n : selected) {
            adj[currLevel][docId].insert(n);
            adj[currLevel][n].insert(docId);

            // enforce degree limits
            size_t maxDeg = (currLevel == 0 ? Mmax0 : M);
            if (adj[currLevel][n].size() > maxDeg) {
                pruneNeighbors(currLevel, n, maxDeg);
            }
        }

        // entry point for search in next layer
        if (!selected.empty()) {
            ep = closestCandidate(selected, docEmb);
        }
    }
}

vector<pair<float, Document>> HNSWIndex::searchHNSW(const Document& doc, size_t k) const {
    // Handle empty graph
    if (entryPoint == SIZE_MAX || collection->size() == 0) {
        return vector<pair<float, Document>>();
    }
    
    size_t ep = entryPoint;
    size_t ef = std::max(efSearch, k);
    ep = greedyDescent(0, doc.getEmbedding());
    auto cmpMin = [](const auto& a, const auto& b){return a.first > b.first;};
    auto cmpMax = [](const auto& a, const auto& b){return a.first < b.first;};
    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, decltype(cmpMax)> selected(cmpMax);
    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, decltype(cmpMin)> candidates(cmpMin);
    unordered_set<size_t> visited;
    auto epDoc = collection->getDocument(ep);
    float d = similarity->compute(epDoc.getEmbedding(), doc.getEmbedding());
    if (epDoc.matchesQuery(doc.getMetadata())) {
        selected.push({d, ep});
    }
    candidates.push({d, ep});
    visited.insert(ep);

    while (!candidates.empty()) {
        auto [dist, node] = candidates.top(); candidates.pop();
        for (auto& it: adj[0].at(node)){
            if (visited.count(it)) continue;
            visited.insert(it);
            auto& itDoc = collection->getDocument(it);
            d = similarity->compute(itDoc.getEmbedding(), doc.getEmbedding());
            if (selected.size() < ef || d > selected.top().first) {
                candidates.push({d, it});
                if (itDoc.matchesQuery(doc.getMetadata())) {
                    selected.push({d, it});
                    if (selected.size() > ef) selected.pop();
                }
            }
        }
    }

    // extract top-k from best-so-far
    k = std::min(selected.size(), k);
    vector<pair<float, Document>> result(k);
    for (size_t i = 0; i < k; i++) {
        auto [score, docId] = selected.top(); selected.pop();
        result[i] = {score, collection->getDocument(docId)};
    }
    return result;
}

void HNSWIndex::updateHNSW(size_t id, Document& doc) {
    if (!collection) {
        throw std::runtime_error("collection is null");
    }
    if (id >= collection->size()) {
        throw std::out_of_range("Document ID out of range");
    }
    
    // Handle empty graph case
    if (maxLevel == SIZE_MAX) {
        // Graph is empty, just insert normally
        insertHNSW(id, doc);
        return;
    }
    
    // clear this id's node on every level
    for (int i = static_cast<int>(maxLevel); i >= 0; i--) {
        adj[i].erase(id);
        if (adj[i].empty()) {
            adj.pop_back();
            if (maxLevel > 0) {
                maxLevel--;
            } else {
                maxLevel = SIZE_MAX; // Mark as empty
                entryPoint = SIZE_MAX;
                break;
            }
        }
    }
    // insert the new document into the HNSW graph
    insertHNSW(id, doc);
}

HNSWIndex::HNSWIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
:Index(coll, sim){}

size_t HNSWIndex::insert(Document& doc){
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    size_t docId = collection->insert(doc);
    insertHNSW(docId, doc);
    return docId;
}

void HNSWIndex::update(size_t id, Document& doc) {
    if (!collection) {
        throw std::runtime_error("collection is null");
    }
    collection->update(id, doc);
    updateHNSW(id, doc);
}

// Method to search for documents of a specific metadata
vector<Document> HNSWIndex::search(const Metadata& meta, size_t k) const {
    vector<Document> results;
    // Iterate through the collection and find documents matching the metadata
    for (const auto& doc : collection->getDocuments()) {
        if (doc.getMetadata().contains(meta)) {
            results.emplace_back(doc);
            if (results.size() >= k) {
                break; // Stop if we have enough results
            }
        }
    }
    return results;
}

// Method to search for similar documents by embedding
vector<Document> HNSWIndex::search(const vector<float>& embedding, size_t k) const {
    if (!collection) {
        throw std::runtime_error("collection is null");
    }
    auto temp = searchHNSW(Document(embedding), k);
    vector<Document> results;
    for (const auto& [score, doc] : temp) {
        results.push_back(doc);
    }
    return results;
}

// Method to search for similar documents and their scores by embedding
vector<pair<float, Document>> HNSWIndex::searchWithScores(const vector<float>& embedding, size_t k) const {
    if (!collection) {
        throw std::runtime_error("collection is null");
    }
    return searchHNSW(Document(embedding), k);
}


vector<Document> HNSWIndex::search(const Metadata& meta, const vector<float>& embedding, size_t k) const {
    if (!collection) {
        throw std::runtime_error("collection is null");
    }
    auto temp = searchHNSW(Document(embedding, meta), k);
    vector<Document> results;
    for (const auto& [score, doc] : temp) {
        results.push_back(doc);
    }
    return results;
}

vector<pair<float, Document>> HNSWIndex::searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const {
    if (!collection) {
        throw std::runtime_error("collection is null");
    }
    return searchHNSW(Document(embedding, meta), k);
}