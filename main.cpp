#include "include/vector.hpp"
#include "include/vector_store.hpp"
#include "include/similarity.hpp"
#include "include/search_engine.hpp"
#include <iostream>
#include <variant>
// namespace
using std::unordered_map;
using std::string;
using std::vector;
using std::cout;
using std::visit;
using std::variant;

void printMetadata(const unordered_map<string, variant<string, int>>& metadata) {
    for (const auto& [key, value] : metadata) {
        cout << key << ": ";
        visit([](auto&& val) { cout << val; }, value);
        cout << ", ";
    }
    cout << "\n";
}

int main() {
    VectorStore store;
    CosineSimilarity similarity;
    SearchEngine engine(store, similarity);

    // Insert sample vectors
    store.insert(Vector({1.0f, 0.0f}, {{"id", 1}, {"type", string("A")}}));
    store.insert(Vector({0.0f, 1.0f}, {{"id", 2}, {"type", string("B")}}));
    store.insert(Vector({0.7f, 0.7f}, {{"id", 3}, {"type", string("C")}}));

    // Query vector
    vector<float> query = {1.0f, 1.0f};

    auto results = engine.search(query, 2);

    cout << "Top 2 matches:\n";
    for (const auto& [score, vec] : results) {
        cout << "Score: " << score << " | Metadata: ";
        printMetadata(vec.metadata);
    }

    return 0;
}
