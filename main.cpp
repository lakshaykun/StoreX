#include "document.hpp"
#include "vector_store.hpp"
#include "similarity.hpp"
#include "search_engine.hpp"
#include "metadata.hpp"
#include <iostream>
#include <variant>
// namespace
using std::unordered_map;
using std::string;
using std::vector;
using std::cout;
using std::visit;
using std::variant;

void printMetadata(const Metadata& metadata) {
    for (const auto& [key, value] : metadata) {
        cout << key << ": ";
        visit([](auto&& val) { cout << val; }, value);
        cout << ", ";
    }
    cout << "\n";
}

int main() {
    // Create VectorStore with storage functionality
    VectorStore store("documents.jsonl", true);  // Enable auto-save
    DotProductSimilarity similarity;
    FlatSearchEngine engine(store, similarity);

    cout << "Loaded " << store.size() << " documents from storage\n";

    // Insert sample vectors (will be auto-saved)
    store.insert(Document({1.0f, 0.0f}, {{"id", 1}, {"type", string("A")}}));
    store.insert(Document({0.5f, 1.0f}, {{"id", 2}, {"type", string("B")}}));
    store.insert(Document({1.0f, 1.0f}, {{"id", 3}, {"type", string("C")}}));
    store.insert(Document({1.6f, 0.3f}, {{"id", 4}, {"type", string("A")}}));
    store.insert(Document({0.5f, 0.8f}, {{"id", 5}, {"type", string("A")}}));
    store.insert(Document({1.6f, 0.3f}, {{"id", 6}, {"class", 5}, {"type", string("A")}}));
    store.insert(Document({0.5f, 0.8f}, {{"id", 7}, {"class", 4}}));

    cout << "Total documents in store: " << store.size() << "\n";

    // Query vector
    vector<float> query = {1.0f, 1.0f};
    // json f = {
    //     {"field", "type"},
    //     {"op", "EQ"},
    //     {"value", "A"}
    // };
    // string raw = R"(
    //     {
    //     "op": "OR",
    //     "children": [
    //         { "field": "type", "op": "EQ", "value": "A" },
    //         { "field": "class", "op": "EQ", "value": 5 }
    //     ]
    //     }
    // )";
    string raw = R"(
    {
        "op": "NEQ",
        "field": "class",
        "value": "4"
    }
    )";
    json f = json::parse(raw);
    auto results = engine.search(query, 2, f);
    // auto results = engine.search(query, 2);

    cout << "Top 2 matches:\n";
    for (const auto& [score, doc] : results) {
        cout << "Score: " << score << " | Metadata: ";
        printMetadata(doc.metadata);
    }

    return 0;
}
