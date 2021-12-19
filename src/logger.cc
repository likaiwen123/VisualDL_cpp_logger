#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "web_logger.h"

using std::cerr;
using std::endl;
using std::ifstream;
using std::ostringstream;
using std::string;

string read_binary_file(const string &filename) {
    ostringstream ss;
    ifstream fin(filename, std::ios::binary);
    if (!fin) {
        cerr << "failed to open file " << filename << endl;
        return "";
    }
    ss << fin.rdbuf();
    fin.close();
    return ss.str();
}

string get_parent_dir(const string &path) {
    auto last_slash_pos = path.find_last_of("/\\");
    if (last_slash_pos == string::npos) {
        return "./";
    }
    return path.substr(0, last_slash_pos + 1);
}
