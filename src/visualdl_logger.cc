#include <google/protobuf/text_format.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "record.pb.h"
#include "web_logger.h"

using std::endl;
using std::ifstream;
using std::numeric_limits;
using std::ofstream;
using std::ostringstream;
using std::string;
using std::to_string;
using std::vector;

using google::protobuf::TextFormat;

using visualdl::Record;
using visualdl::Record_Audio;
using visualdl::Record_bytes_embeddings;
using visualdl::Record_Embedding;
using visualdl::Record_Embeddings;
using visualdl::Record_Histogram;
using visualdl::Record_HParam;
using visualdl::Record_HParam_HparamInfo;
using visualdl::Record_Image;
using visualdl::Record_MetaData;
using visualdl::Record_PRCurve;
using visualdl::Record_ROC_Curve;
using visualdl::Record_Text;
using visualdl::Record_Value;

int TensorBoardLogger::add_scalar(const string &tag, int step, double value,
                                  time_t walltime) {
    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }
    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(step);
    v->set_tag(tag);
    v->set_timestamp(walltime);
    v->set_value(static_cast<float>(value));

    return add_record(record);
}

int TensorBoardLogger::add_meta(const std::string &tag,
                                const std::string &display_name, int64_t step,
                                time_t timestamp) {
    if (timestamp < 0) {
        timestamp = time(nullptr) * 1000;
    }

    auto *meta = new Record_MetaData();
    meta->set_display_name(display_name);

    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(step);
    v->set_tag(tag);
    v->set_timestamp(timestamp);
    v->set_allocated_meta_data(meta);

    return add_record(record);
}

int TensorBoardLogger::add_image(const std::string &tag, int step,
                                 const std::string &encoded_image,
                                 time_t walltime) {
    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }

    auto *image = new Record_Image();
    image->set_encoded_image_string(encoded_image);

    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(step);
    v->set_tag(tag);
    v->set_timestamp(walltime);
    v->set_allocated_image(image);

    return add_record(record);
}

int TensorBoardLogger::add_image_from_path(const std::string &tag, int step,
                                           const std::string &path,
                                           time_t walltime) {
    return add_image(tag, step, read_binary_file(path), walltime);
}

int TensorBoardLogger::write(Record &record) {
    string buf;
    record.SerializeToString(&buf);
    auto buf_len = static_cast<uint64_t>(buf.size());

    ofs_->write((char *)&buf_len, sizeof(buf_len));
    ofs_->write(buf.c_str(), buf.size());

    ofs_->flush();
    return 0;
}
