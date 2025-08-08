#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

// Include your StoreX headers
#include "../include/vector_store.hpp"
#include "../include/document.hpp"
#include "../include/search_engine.hpp"
#include "../include/similarity.hpp"
#include "../include/metadata.hpp"
#include "../include/json.hpp"

namespace py = pybind11;

// Simple helper function to convert Python dict to Metadata
Metadata dict_to_metadata(const py::dict& dict) {
    Metadata metadata;
    for (auto item : dict) {
        std::string key = py::str(item.first);
        
        if (py::isinstance<py::str>(item.second)) {
            metadata[key] = std::string(py::str(item.second));
        } else if (py::isinstance<py::int_>(item.second)) {
            metadata[key] = item.second.cast<int>();
        } else if (py::isinstance<py::float_>(item.second)) {
            metadata[key] = item.second.cast<float>();
        }
    }
    return metadata;
}

// Helper function to convert Metadata to Python dict
py::dict metadata_to_dict(const Metadata& metadata) {
    py::dict dict;
    for (const auto& pair : metadata) {
        const std::string& key = pair.first;
        const MetadataValue& value = pair.second;
        
        std::visit([&](const auto& v) {
            dict[key.c_str()] = v;
        }, value);
    }
    return dict;
}

PYBIND11_MODULE(storex, m) {
    m.doc() = "StoreX: High-performance vector database for ML workflows";
    
    // Document class
    py::class_<Document>(m, "Document")
        .def(py::init<>(), "Create empty document")
        .def(py::init<const std::vector<float>&>(), "Create document with embedding")
        .def(py::init<const std::vector<float>&, const Metadata&>(), 
             "Create document with embedding and metadata")
        .def(py::init([](const std::vector<float>& embedding, const py::dict& metadata_dict) {
            return Document(embedding, dict_to_metadata(metadata_dict));
        }), "Create document with embedding and metadata dict")
        .def_readwrite("embedding", &Document::embedding, "Vector embedding")
        .def_property("metadata", 
            [](const Document& doc) { return metadata_to_dict(doc.metadata); },
            [](Document& doc, const py::dict& dict) { doc.metadata = dict_to_metadata(dict); },
            "Document metadata as dictionary")
        .def("__repr__", [](const Document& doc) {
            return "<Document embedding_dim=" + std::to_string(doc.embedding.size()) + 
                   " metadata_keys=" + std::to_string(doc.metadata.size()) + ">";
        });
    
    // VectorStore class
    py::class_<VectorStore>(m, "VectorStore")
        .def(py::init<>(), "Create vector store without persistence")
        .def(py::init<const std::string&, bool>(), 
             "Create vector store with storage path",
             py::arg("storage_path"), py::arg("auto_save") = true)
        .def("insert", py::overload_cast<const Document&>(&VectorStore::insert),
             "Insert a single document")
        .def("insert", py::overload_cast<const std::vector<Document>&>(&VectorStore::insert),
             "Insert multiple documents")
        .def("insert", [](VectorStore& self, const std::vector<float>& embedding, const py::dict& metadata_dict) {
            Document doc(embedding, dict_to_metadata(metadata_dict));
            self.insert(doc);
        }, "Insert document from embedding and metadata dict",
           py::arg("embedding"), py::arg("metadata") = py::dict())
        .def("get_all", &VectorStore::getAll, 
             "Get all documents", py::return_value_policy::reference_internal)
        .def("save", &VectorStore::save, "Save to storage")
        .def("load", &VectorStore::load, "Load from storage")
        .def("clear", &VectorStore::clear, "Clear all documents")
        .def("size", &VectorStore::size, "Get number of documents")
        .def("__len__", &VectorStore::size)
        .def("has_storage", &VectorStore::hasStorage, "Check if storage is enabled")
        .def("__repr__", [](const VectorStore& store) {
            return "<VectorStore size=" + std::to_string(store.size()) + 
                   " has_storage=" + (store.hasStorage() ? "true" : "false") + ">";
        });
    
    // Similarity metrics
    py::class_<SimilarityMetric>(m, "SimilarityMetric")
        .def("compute", &SimilarityMetric::compute, 
             "Compute similarity between two vectors",
             py::arg("a"), py::arg("b"));
    
    py::class_<CosineSimilarity, SimilarityMetric>(m, "CosineSimilarity")
        .def(py::init<>());
    
    py::class_<DotProductSimilarity, SimilarityMetric>(m, "DotProductSimilarity")
        .def(py::init<>());
    
    py::class_<EuclideanSimilarity, SimilarityMetric>(m, "EuclideanSimilarity")
        .def(py::init<>());
    
    // Search engines - simplified without json filter for now
    py::class_<SearchEngine>(m, "SearchEngine")
        .def("search", [](const SearchEngine& self, const std::vector<float>& query, size_t k) {
            // Use empty json filter for now
            json filter = json{};
            return self.search(query, k, filter);
        }, "Search for similar vectors",
           py::arg("query"), py::arg("k") = 5);
    
    py::class_<FlatSearchEngine, SearchEngine>(m, "FlatSearchEngine")
        .def(py::init<const VectorStore&, const SimilarityMetric&>(),
             py::arg("store"), py::arg("metric"));
    
    py::class_<LSHSearchEngine, SearchEngine>(m, "LSHSearchEngine")
        .def(py::init<const VectorStore&, const SimilarityMetric&, size_t, size_t>(),
             py::arg("store"), py::arg("metric"), py::arg("num_tables") = 10, py::arg("num_hashes_per_table") = 8);
    
    // Utility functions for direct similarity computation
    m.def("cosine_similarity", [](const std::vector<float>& a, const std::vector<float>& b) {
        CosineSimilarity metric;
        return metric.compute(a, b);
    }, "Calculate cosine similarity between two vectors");
    
    m.def("dot_product_similarity", [](const std::vector<float>& a, const std::vector<float>& b) {
        DotProductSimilarity metric;
        return metric.compute(a, b);
    }, "Calculate dot product similarity between two vectors");
    
    m.def("euclidean_similarity", [](const std::vector<float>& a, const std::vector<float>& b) {
        EuclideanSimilarity metric;
        return metric.compute(a, b);
    }, "Calculate euclidean similarity between two vectors");
    
    // Version info
    m.attr("__version__") = "0.1.0";
}
