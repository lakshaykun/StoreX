#include "vector_store.hpp"

int main() {
    cout << "starting test\n";
    auto col = make_shared<Collection>();
    auto sim = make_shared<EuclideanSimilarity>();
    auto ind = make_shared<FlatIndex>(col, sim);
    vector_store store(ind, col);
    vector<Document> docs;
    Metadata meta1(string(R"({"title" : "Lakshay", "id" : "1", "genre" : ["horror", "fantasy"]})"));
    Metadata meta2(string(R"({"title" : "Mitali", "id" : "2", "genre" : ["space", "fantasy"]})"));

    for (float i=-10; i<=10; i++){
        for (float j=-10; j<=5; j++){
            for (float k=-1; k<=10; k++){
                docs.emplace_back(
                    Document(
                        {i, j, k}, 
                        meta1
                    )
                );
            }
        }
    }
    for (float i=-5; i<=1; i++){
        for (float j=10; j<=15; j++){
            for (float k=-10; k<=-5; k++){
                docs.emplace_back(
                    Document(
                        {i, j, k}, 
                        meta2
                    )
                ); 
            }
        }
    }

    vector<size_t> ids = store.insert(docs);
    // for (int i=0; i<10; i++) cout << ids[i] << ' ';
    cout << '\n';
    vector<float> emb = {1.0, 1.0, 1.0};
    // vector<std::pair<float, Document>> res1 = store.searchWithScores(emb, 5);
    // for (auto& it: res1) {
    //     cout << "Score: " << it.first << " ";
    //     it.second.printDocument();
    // }
    // cout << "searching with metadata1\n";
    // vector<std::pair<float, Document>> res2 = store.searchWithScores(meta1, emb, 5);
    // for (auto& it: res2) {
    //     cout << "Score: " << it.first << " ";
    //     it.second.printDocument();
    // }
    // cout << "searching with metadata2\n";
    // vector<std::pair<float, Document>> res3 = store.searchWithScores(meta2, emb, 5);
    // for (auto& it: res3) {
    //     cout << "Score: " << it.first << " ";
    //     it.second.printDocument();
    // }

    // cout << "searching without score for metadata2\n";
    // vector<Document> res4 = store.search(meta2, emb, 5);
    // for (auto& it: res4) {
    //     it.printDocument();
    // }

    cout << store.fetchId(store.fetchDocument(1)) << '\n';

    return 0;
}