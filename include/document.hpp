#pragma once
#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>
using std::vector;
using std::unordered_map;
using std::string;
using std::cout;
using json = nlohmann::json;

class Metadata {
private:
    // JSON object to store metadata
    json data;
public:
    // default constructor
    Metadata () { data = json::object(); }
    // constructor from json object
    Metadata(const json& j) : data(j) {}
    // constructor from JSON string
    Metadata(const string& jsonString) {
        // if the string is empty, initialize with an empty JSON object
        if (jsonString.empty()) {
            data = json::object();
            return;
        }
        // parse the JSON string
        try {
            data = json::parse(jsonString);
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing JSON string: " << e.what() << std::endl;
            data = json::object();
        }
    }

    // method to convert Metadata to JSON string
    void printMetadata() const {
        std::cout << data.dump(4) << std::endl;
    }

    // method to get the data
    const json& getData() const {
        return data;
    }

    // Method to get the data as a string
    string toString() const {
        return data.dump();
    }

    // operator to compare Metadata objects
    bool operator==(const Metadata& other) const {
        return data == other.getData();
    }

    // Method to check if this metadata contains all key-value pairs from query metadata
    bool contains(const Metadata& queryMetadata) const {
        const json& queryData = queryMetadata.getData();
        
        // Iterate through all key-value pairs in the query metadata
        for (auto it = queryData.begin(); it != queryData.end(); ++it) {
            const string& key = it.key();
            const json& queryValue = it.value();
            
            // Check if the key exists in this metadata
            if (data.find(key) == data.end()) {
                return false;
            }
            
            // Check if the values match
            if (data[key] != queryValue) {
                return false;
            }
        }
        
        return true;
    }
};

class Document {
private:
    vector<float> embedding;
    Metadata metadata;
public:
    // default constructor
    Document() = default;
    
    // constructor with embedding only
    Document(vector<float> emb) : embedding(std::move(emb)) {
        // Initialize metadata with an empty JSON object
        metadata = Metadata(json::object());
    }

    // constructor with embedding and metadata
    Document(vector<float> emb, const Metadata& meta) 
        : embedding(std::move(emb)), metadata(meta) {}
    
    // constructor from JSON object
    Document(vector<float> emb, const json& j)
        : embedding(std::move(emb)), metadata(j) {}

    // constructor from JSON string
    Document(vector<float> emb, const string& jsonString)
        : embedding(std::move(emb)), metadata(jsonString) {}

    // method to convert Document to JSON string
    void printDocument() const {
        json j;
        j["embedding"] = embedding;
        j["metadata"] = metadata.getData();
        cout << j.dump(4) << std::endl;
    }

    // method to get the embedding
    const vector<float>& getEmbedding() const {
        return embedding;
    }

    // method to get the metadata
    const Metadata& getMetadata() const {
        return metadata;
    }

    // method to set the embedding
    void setEmbedding(const vector<float>& emb) {
        embedding = emb;
    }

    // method to set the metadata
    void setMetadata(const Metadata& meta) {
        metadata = meta;
    }

    // method to check if the document is empty
    bool isEmpty() const {
        return embedding.empty() && metadata.getData().empty();
    }

    // operator to compare Document objects
    bool operator==(const Document& other) const {
        return embedding == other.getEmbedding() && metadata == other.getMetadata();
    }
    
    bool operator!=(const Document& other) const {
        return !(*this == other);
    }

    // Method to check if document metadata contains query metadata
    bool matchesQuery(const Metadata& queryMetadata) const {
        return metadata.contains(queryMetadata);
    }

    // Overloaded version that accepts JSON object
    bool matchesQuery(const json& queryJson) const {
        Metadata queryMetadata(queryJson);
        return metadata.contains(queryMetadata);
    }

    // Overloaded version that accepts JSON string
    bool matchesQuery(const string& queryJsonString) const {
        Metadata queryMetadata(queryJsonString);
        return metadata.contains(queryMetadata);
    }
};