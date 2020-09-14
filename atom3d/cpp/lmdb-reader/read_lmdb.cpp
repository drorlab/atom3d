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


const string LMDB_FILE = "../../../data/dataset_tests/json_lmdb";


string gzip_decompress(const char *ptr, size_t size) {
    // Decompress returns a string and decodes both zlib and gzip
    return gzip::decompress(ptr, size);
}

json parse_json(const string &str) {
    json j = json::parse(str);
    return j;
}

void read_lmdb(const string &filename, const lmdb::val &key) {
    // Create and open the LMDB environment
    auto env = lmdb::env::create();
    env.open(filename.c_str());

    {
        auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
        auto dbi = lmdb::dbi::open(rtxn, nullptr);

        lmdb::val val;
        if (lmdb::dbi_get(rtxn, dbi, key, val)) {
            json j = parse_json(gzip_decompress(val.data(), val.size()));
            cout << "Key: " << key << ", val: " << j.dump(4) << "\n" << endl;
        } else {
            cerr << "ERROR: key " << key.data() << " not found in the database" << endl;
        }
    } // rtxn aborted automatically*/

    // The enviroment is closed automatically. */
}

int main() {
    cout << "Reading LMDB file " << LMDB_FILE << endl;
    read_lmdb(LMDB_FILE, "0");
    cout << "Done" << endl;

    return EXIT_SUCCESS;
}
