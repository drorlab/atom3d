#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gzip/compress.hpp>
#include <gzip/decompress.hpp>
#include <gzip/utils.hpp>

#include <nlohmann/json.hpp>

#include <lmdb++.h>

using namespace std;
using json = nlohmann::json;


const string LMDB_FILE = "../../data/dataset_tests/json_lmdb";


string gzip_decompress(const string &compressed_data) {
    // Decompress returns a string and decodes both zlib and gzip
    const char *compressed_pointer = compressed_data.data();
    string decompressed_data = gzip::decompress(compressed_pointer, compressed_data.size());
    return decompressed_data;
}

string read_gzip(const string &filename) {
    ifstream ifs(filename, ios::in | ios::binary);
    stringstream buffer;
    buffer << ifs.rdbuf();
    string compressed_data = buffer.str();
    return gzip_decompress(compressed_data);
}

json parse_json(const string &str) {
    json j = json::parse(str);
    return j;
}

string read_lmdb(const string &filename) {
    // Create and open the LMDB environment
    auto env = lmdb::env::create();
    env.open(filename.c_str());

    // Fetch key/value pairs in a read-only transaction
    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
    auto dbi = lmdb::dbi::open(rtxn, nullptr);

    auto cursor = lmdb::cursor::open(rtxn, dbi);
    string key, value;
    int i = 0;
    while (cursor.get(key, value, MDB_NEXT) && i < 2) {
        json j = json::parse(gzip_decompress(value));
        cout << "Key: " << key << ", val: " << j.dump(4) << "\n" << endl;
        i++;
    }
    cursor.close();

    /*
    TODO (psuriana): Make this to work
    std::string value;
    bool fail = dbi.get(rtxn, key, value);
    if (!fail) {
        json j = json::parse(gzip_decompress(value));
        cout << "Key: " << key << ", val: " << j.dump(4) << "\n" << endl;
    } else {
        cout << "ERROR: key " << key << " not found in the database" << endl;
    }*/

    rtxn.abort();
    return value;
    // The enviroment is closed automatically. */
}

int main() {
    cout << "Reading LMDB file " << LMDB_FILE << endl;
    read_lmdb(LMDB_FILE);
    cout << "Done" << endl;

    return EXIT_SUCCESS;
}
