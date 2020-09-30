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


//const string LMDB_FILE = "../../../data/dataset_tests/json_lmdb";
const string LMDB_FILE = "/mnt/d/atom3d/data/qm9/lmdb";

string gzip_decompress(const char *ptr, size_t size) {
    // Decompress returns a string and decodes both zlib and gzip
    return gzip::decompress(ptr, size);
}

json parse_json(const string &str) {
    json j = json::parse(str);
    return j;
}

int find_index(json jlist, const string s) {
    unsigned int i = 0;
    while (i < jlist.size()) {
	if (jlist[i] == s) {
	    break;
	}
	i++;
    }
    return i;
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
	    // Extract the various entries from the JSON object
            json id = j["/id"_json_pointer];
	    json columns = j["/atoms/columns"_json_pointer];
	    json data = j["/atoms/data"_json_pointer];
	    json index = j["/atoms/index"_json_pointer];
	    // Print general info 
	    cout << key << ", " << id << ", " << data.size() << " atoms" << endl;
	    cout << columns.size() << " columns: " << columns << endl;
            // Declare the vectors
	    vector <int> model (data.size());
	    vector <string> chain (data.size());
	    vector <string> hetero (data.size());
	    vector <string> insertion_code (data.size());
	    vector <int> residue (data.size());
	    vector <string> segid (data.size());
	    vector <string> resname (data.size());
	    vector <string> altloc (data.size());
	    vector <float> occupancy (data.size());
	    vector <float> bfactor (data.size());
	    vector< vector <double> > coordinates ( data.size(), vector<double> (3, 0));
            vector <string> element (data.size());
	    vector <string> name (data.size());
	    vector <string> fullname (data.size());
            vector <int> serial_number (data.size());
	    // Loop through atoms
	    int ac = 0;
	    for (auto& atom : data.items()) {
		// Print the info about the atom
		cout << atom.value() << '\n'; // x.key() provides index
                // Fill the vectors with data
		model[ac] = atom.value()[find_index(columns,"model")];
		chain[ac] = atom.value()[find_index(columns,"chain")];
		hetero[ac] = atom.value()[find_index(columns,"hetero")];
		insertion_code[ac] = atom.value()[find_index(columns,"insertion_code")];
		residue[ac] = atom.value()[find_index(columns,"residue")];
		segid[ac] = atom.value()[find_index(columns,"segid")];
		resname[ac] = atom.value()[find_index(columns,"resname")];
		altloc[ac] = atom.value()[find_index(columns,"altloc")];
		occupancy[ac] = atom.value()[find_index(columns,"occupancy")];
		bfactor[ac] = atom.value()[find_index(columns,"bfactor")];
		coordinates[ac][0] = atom.value()[find_index(columns,"x")];
		coordinates[ac][1] = atom.value()[find_index(columns,"y")];
		coordinates[ac][2] = atom.value()[find_index(columns,"z")];
		element[ac] = atom.value()[find_index(columns,"element")];
		name[ac] = atom.value()[find_index(columns,"name")];
		fullname[ac] = atom.value()[find_index(columns,"fullname")];
		serial_number[ac] = atom.value()[find_index(columns,"serial_number")];
		// Increase the counter
		ac++;
            }
	    cout << "Done reading. Starting test output." << endl;
            for (unsigned int i=0; i < element.size(); i++) {
                cout << model[i] << " " 
		     << chain[i] << " " 
		     << hetero[i] << " " 
		     << insertion_code[i] << " " 
		     << residue[i] << " " 
		     << segid[i] << " "
		     << resname[i] << " "
		     << altloc[i] << " "
		     << occupancy[i] << " "
		     << bfactor[i] << " "
		     << coordinates[i][0] << " "
		     << coordinates[i][1] << " "
		     << coordinates[i][2] << " "
		     << element[i] << " "
		     << name[i] << " "
		     << fullname[i] << " "
		     << serial_number[i] << " " << endl;
	    }
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
