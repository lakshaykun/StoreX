#pragma once
#include <sqlite3.h>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <cstring>
#include "document.hpp"

using std::string;

class Storage {
private:
    // Storage implementation details
    sqlite3* db;
    string filename = "db/storex.db";
    const char* createSQL =
        "CREATE TABLE IF NOT EXISTS StoreX ("
        "id INTEGER PRIMARY KEY, "
        "metadata TEXT NOT NULL, "
        "embedding BLOB NOT NULL);";
    const char* insertSQL = "INSERT INTO StoreX (id, metadata, embedding) VALUES (?, ?, ?);";
    const char* updateSQL = "UPDATE StoreX SET metadata = ?, embedding = ? WHERE id = ?;";
    string serializeVector(const vector<float>& vec) {
        std::string blob(vec.size() * sizeof(float), '\0');
        std::memcpy(blob.data(), vec.data(), blob.size());
        return blob;
    }
    vector<float> deserializeVector(const string& blob) {
        vector<float> vec(blob.size() / sizeof(float));
        std::memcpy(vec.data(), blob.data(), blob.size());
        return vec;
    }
public:
    // default constructor
    Storage(){
        // create db folder if folder not created
        std::filesystem::create_directories("db");
        // Open the database with the default filename
        if (sqlite3_open(filename.c_str(), &db) != SQLITE_OK) {
            throw std::runtime_error("Failed to open database");
            return;
        }
        // Database opened successfully
        char* errMsg = nullptr;
        if (sqlite3_exec(db, createSQL, nullptr, nullptr, &errMsg) != SQLITE_OK) {
            string error = errMsg;
            sqlite3_free(errMsg);
            throw std::runtime_error("Error creating table: " + error);
        }
    }

    // constructor with custom filename
    Storage(const string& customFilename) : filename(customFilename) {
        if (sqlite3_open(filename.c_str(), &db) != SQLITE_OK) {
            throw std::runtime_error("Failed to open database");
            return;
        }
        // Database opened successfully
        char* errMsg = nullptr;
        if (sqlite3_exec(db, createSQL, nullptr, nullptr, &errMsg) != SQLITE_OK) {
            string error = errMsg;
            sqlite3_free(errMsg);
            throw std::runtime_error("Error creating table: " + error);
        }
    }

    // destructor
    ~Storage() {
        if (db) {
            sqlite3_close(db);
        }
    }

    // Method to insert a document into the storage
    void insert(size_t id, const Document& doc) {
        if (!db) {
            throw std::runtime_error("Database is not initialized.");
        }
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, insertSQL, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare insert statement");
        }

        // Bind the parameters
        sqlite3_bind_int(stmt, 1, id);
        sqlite3_bind_text(stmt, 2, doc.getMetadata().toString().c_str(), -1, SQLITE_TRANSIENT);
        string serializedEmbedding = serializeVector(doc.getEmbedding());
        sqlite3_bind_blob(stmt, 3, serializedEmbedding.data(), serializedEmbedding.size(), SQLITE_TRANSIENT);

        // Execute the statement
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            throw std::runtime_error("Failed to execute insert statement");
        }

        sqlite3_finalize(stmt);
    }

    // Method to update a document in the storage
    void update(size_t id, const Document& doc) {
        if (!db) {
            throw std::runtime_error("Database is not initialized.");
        }

        // Prepare the update statement
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, updateSQL, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare update statement");
        }

        // Bind the parameters
        sqlite3_bind_text(stmt, 1, doc.getMetadata().toString().c_str(), -1, SQLITE_TRANSIENT);
        string serializedEmbedding = serializeVector(doc.getEmbedding());
        sqlite3_bind_blob(stmt, 2, serializedEmbedding.data(), serializedEmbedding.size(), SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, id);

        // Execute the statement
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            throw std::runtime_error("Failed to execute update statement");
        }

        sqlite3_finalize(stmt);
    }

    // Method to store collection of documents
    void insert(const vector<Document>& docs) {
        if (!db) {
            throw std::runtime_error("Database is not initialized.");
        }
        sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
        for (size_t i = 0; i < docs.size(); ++i) {
            try {
                insert(i, docs[i]);
            } catch (const std::exception& e) {
                sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
                throw std::runtime_error("Error inserting document at index " + std::to_string(i) + ": " + e.what());
            }
        }
        sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
    }

    // load collection of documents
    vector<Document> load() {
        if (!db) {
            throw std::runtime_error("Database is not initialized.");
        }
        vector<Document> docs;
        sqlite3_stmt* stmt;
        const char* selectSQL = "SELECT id, metadata, embedding FROM StoreX ORDER BY id;";
        if (sqlite3_prepare_v2(db, selectSQL, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare select statement");
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            string metadata = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            string embeddingBlob(reinterpret_cast<const char*>(sqlite3_column_blob(stmt, 2)), sqlite3_column_bytes(stmt, 2));
            vector<float> embedding = deserializeVector(embeddingBlob);
            docs.emplace_back(embedding, metadata);
        }

        sqlite3_finalize(stmt);
        return docs;
    }

    // clear the storage
    void clear() {
        if (!db) {
            throw std::runtime_error("Database is not initialized.");
        }
        char* errMsg = nullptr;
        if (sqlite3_exec(db, "DELETE FROM StoreX;", nullptr, nullptr, &errMsg) != SQLITE_OK) {
            string error = errMsg;
            sqlite3_free(errMsg);
            throw std::runtime_error("Error clearing table: " + error);
        }
    }

    // delete the database
    void deleteDatabase() {
        if (db) {
            sqlite3_close(db);
            db = nullptr;
        }
        if (std::filesystem::exists(filename)) {
            std::filesystem::remove(filename);
            std::filesystem::remove("db");
        }
    }
};
