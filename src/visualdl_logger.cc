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
#include "md5.h"

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

int TensorBoardLogger::add_audio(const std::string &tag, int step,
                                 const std::string &encoded_audio,
                                 float sample_rate, time_t walltime) {
    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }

    auto *audio = new Record_Audio();
    audio->set_encoded_audio_string(encoded_audio);
    audio->set_sample_rate(sample_rate);

    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(step);
    v->set_tag(tag);
    v->set_timestamp(walltime);
    v->set_allocated_audio(audio);

    return add_record(record);
}

int TensorBoardLogger::add_audio_from_path(const std::string &tag, int step,
                                           const std::string &path,
                                           float sample_rate, time_t walltime) {
    return add_audio(tag, step, read_binary_file(path), sample_rate, walltime);
}

int TensorBoardLogger::add_text(const std::string &tag, int step,
                                const std::string &text, time_t walltime) {
    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }

    auto *_text = new Record_Text();
    _text->set_encoded_text_string(text);

    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(step);
    v->set_tag(tag);
    v->set_timestamp(walltime);
    v->set_allocated_text(_text);

    return add_record(record);
}

int TensorBoardLogger::add_embeddings(
    const std::string &tag, const std::vector<std::vector<float>> &mat,
    const std::vector<std::string> &metadata,
    const std::vector<std::string> &metadata_header, time_t walltime) {
    vector<vector<string>> meta(1, metadata);
    return add_embeddings(tag, mat, meta, metadata_header, walltime);
}

int TensorBoardLogger::add_embeddings(
    const std::string &tag, const std::vector<std::vector<float>> &mat,
    const std::vector<std::vector<std::string>> &metadata,
    const std::vector<std::string> &metadata_header, time_t walltime) {
    assert(!metadata.empty());
    assert(mat.size() == metadata[0].size());

    std::vector<std::string> header;

    if (metadata_header.empty()) {
        if (metadata.size() > 1) {
            header.resize(metadata.size());
            for (size_t i = 0; i < metadata.size(); ++i) {
                header[i] = "label_" + to_string(i);
            }
        }
    } else {
        assert(metadata.size() == metadata_header.size());
    }

    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }

    auto *embs = new Record_Embeddings();

    for (const auto &meta : metadata_header) {
        embs->add_label_meta(meta);
    }
    for (const auto &meta : header) {
        embs->add_label_meta(meta);
    }
    for (size_t i = 0; i < mat.size(); ++i) {
        auto emb = embs->add_embeddings();
        for (const auto &meta : metadata) {
            emb->add_label(meta[i]);
        }
        for (const auto &v : mat[i]) {
            emb->add_vectors(v);
        }
    }

    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(0);
    v->set_tag(tag);
    v->set_timestamp(walltime);
    v->set_allocated_embeddings(embs);

    return add_record(record);
}

int TensorBoardLogger::add_hparams(
    const std::map<std::string, std::string> &hparams_dict,
    const std::vector<std::string> &metrics_list, time_t walltime) {
    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }

    string name = md5(log_file_);

    auto *hparams = new Record_HParam();
    hparams->set_name(name);

    for (const auto &pair : hparams_dict) {
        const auto &k = pair.first;
        const auto &v = pair.second;

        auto *hparam_info = hparams->add_hparaminfos();
        hparam_info->set_name(k);

        int64_t v_int;
        double v_float;
        try {
            // todo: stoi returns an int, not int64
            size_t p = 0;
            v_int = std::stoi(v, &p);
            if (p == v.length()) {
                hparam_info->set_int_value(v_int);
            }
            else {
                throw std::invalid_argument("\"" + v + "\" is not an integer.");
            }
        } catch (const std::invalid_argument &) {
            try {
                size_t p = 0;
                v_float = std::stod(v, &p);
                if (p == v.length()) {
                    hparam_info->set_float_value(v_float);
                }
                else {
                    throw std::invalid_argument("\"" + v + "\" is not a double.");
                }
            } catch (const std::invalid_argument &) {
                hparam_info->set_string_value(v);
            }
        }
    }

    for (const auto &v : metrics_list) {
        auto *metric_info = hparams->add_metricinfos();
        metric_info->set_name(v);
        metric_info->set_float_value(0);
    }

    auto *record = new Record();
    auto v = record->add_values();
    v->set_id(1);
    v->set_tag("hparam");
    v->set_timestamp(walltime);
    v->set_allocated_hparam(hparams);

    return add_record(record);
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
