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
    Metadata () = default;
    // constructor from json object
    Metadata(const json& j) : data(j) {}
    // constructor from JSON string
    Metadata(const string& jsonString) {
        data = json::parse(jsonString);
    }
    // method to convert Metadata to JSON string
    void printMetadata() const {
        cout << data.dump(4) << std::endl;
    }
    // method to get the data
    const json& getData() const {
        return data;
    }
    // operator to compare Metadata objects
    bool operator==(const Metadata& other) const {
        return data == other.getData();
    }
};

class Document {
private:
    vector<float> embedding;
    Metadata metadata;
public:
    // default constructor
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
};