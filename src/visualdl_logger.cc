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

#include "md5.h"
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
        if (metadata.size() == 1) {
            std::cout << "warning! meataheader should be empty when metadata "
                         "is 1-dim, otherwise VisualDL may fail (tested with "
                         "version 2.2.2), this may be a bug in VisualDL."
                      << std::endl;
        }
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
            } else {
                throw std::invalid_argument("\"" + v + "\" is not an integer.");
            }
        } catch (const std::invalid_argument &) {
            try {
                size_t p = 0;
                v_float = std::stod(v, &p);
                if (p == v.length()) {
                    hparam_info->set_float_value(v_float);
                } else {
                    throw std::invalid_argument("\"" + v +
                                                "\" is not a double.");
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

// todo: merge with calculate_hist_bins in web_logger.h
vector<int> calc_hist(const vector<int> &values, const vector<double> &labels,
                      double weights, int bins, double upper,
                      double lower = 0.0) {
    vector<double> v(values.size(), weights);
    for (size_t i = 0; i < values.size(); ++i) {
        v[i] *= double(values[i]) * labels[i];
    }
    vector<double> bounds(bins + 1);
    double interval = (upper - lower) / bins;
    for (size_t i = 0; i < bins + 1; ++i) {
        bounds[i] = i * interval + lower;
    }
    vector<int> count(bins, 0);
    for (auto t : v) {
        auto pos = std::lower_bound(bounds.begin(), bounds.end(), t);
        if (pos != bounds.end()) {
            // todo: ignore values larger than upper.
            count[pos - bounds.begin()] += 1;
        }
    }
    return count;
}

int TensorBoardLogger::add_pr_curve(const std::string &tag,
                                    const std::vector<double> &labels,
                                    const std::vector<double> &predictions,
                                    int step, int num_thresholds,
                                    time_t walltime, double weights) {
    return add_curve("pr_curve", tag, labels, predictions, step, num_thresholds,
                     walltime, weights);
}

int TensorBoardLogger::add_roc_curve(const std::string &tag,
                                     const std::vector<double> &labels,
                                     const std::vector<double> &predictions,
                                     int step, int num_thresholds,
                                     time_t walltime, double weights) {
    return add_curve("roc_curve", tag, labels, predictions, step,
                     num_thresholds, walltime, weights);
}

int TensorBoardLogger::add_curve(const std::string &type,
                                 const std::string &tag,
                                 const std::vector<double> &labels,
                                 const std::vector<double> &predictions,
                                 int step, int num_thresholds, time_t walltime,
                                 double weights) {
    if (walltime < 0) {
        walltime = time(nullptr) * 1000;
    }

    // todo: parameter validation
    if (num_thresholds > 127) {
        std::cout
            << "warning, num_thresholds can not be larger than 127, set as 127."
            << endl;
        num_thresholds = 127;
    }

    // todo: int64_
    vector<int> bucket_indices(predictions.size(), 0);
    for (size_t i = 0; i < predictions.size(); ++i) {
        bucket_indices[i] =
            static_cast<int>(floor(predictions[i] * (num_thresholds - 1)));
    }

    auto tp_buckets = calc_hist(bucket_indices, labels, weights, num_thresholds,
                                num_thresholds - 1);

    vector<double> neg_labels(labels.size(), 0.0);
    for (size_t i = 0; i < labels.size(); ++i) {
        neg_labels[i] = 1.0 - labels[i];
    }
    auto fp_buckets = calc_hist(bucket_indices, neg_labels, weights,
                                num_thresholds, num_thresholds - 1);

    vector<int> tp(num_thresholds, 0);
    vector<int> fp(num_thresholds, 0);
    vector<int> tn(num_thresholds, 0);
    vector<int> fn(num_thresholds, 0);

    std::reverse(tp_buckets.begin(), tp_buckets.end());
    std::reverse(fp_buckets.begin(), fp_buckets.end());
    std::partial_sum(tp_buckets.begin(), tp_buckets.end(), tp.begin());
    std::partial_sum(fp_buckets.begin(), fp_buckets.end(), fp.begin());
    std::reverse(tp.begin(), tp.end());
    std::reverse(fp.begin(), fp.end());

    const double _MINIMUM_COUNT = 1e-7;

    for (size_t i = 0; i < num_thresholds; ++i) {
        tn[i] = fp[0] - fp[i];
        fn[i] = tp[0] - tp[i];
    }

    if (type == "pr_curve") {
        vector<double> precision(num_thresholds, 0.0);
        vector<double> recall(num_thresholds, 0.0);
        auto *pr_curve = new Record_PRCurve();
        for (size_t i = 0; i < num_thresholds; ++i) {
            precision[i] =
                double(tp[i]) / std::max(_MINIMUM_COUNT, double(tp[i] + fp[i]));
            recall[i] =
                double(tp[i]) / std::max(_MINIMUM_COUNT, double(tp[i] + fn[i]));
            pr_curve->add_tp(tp[i]);
            pr_curve->add_fp(fp[i]);
            pr_curve->add_tn(tn[i]);
            pr_curve->add_fn(fn[i]);
            pr_curve->add_precision(precision[i]);
            pr_curve->add_recall(recall[i]);
        }

        auto *record = new Record();
        auto v = record->add_values();
        v->set_id(step);
        v->set_tag(tag);
        v->set_timestamp(walltime);
        v->set_allocated_pr_curve(pr_curve);

        return add_record(record);
    } else if (type == "roc_curve") {
        vector<double> tpr(num_thresholds, 0.0);
        vector<double> fpr(num_thresholds, 0.0);
        auto *roc_curve = new Record_ROC_Curve();
        for (size_t i = 0; i < num_thresholds; ++i) {
            tpr[i] =
                double(tp[i]) / std::max(_MINIMUM_COUNT, double(tn[i] + fp[i]));
            fpr[i] =
                double(fp[i]) / std::max(_MINIMUM_COUNT, double(tn[i] + fp[i]));
            roc_curve->add_tp(tp[i]);
            roc_curve->add_fp(fp[i]);
            roc_curve->add_tn(tn[i]);
            roc_curve->add_fn(fn[i]);
            roc_curve->add_tpr(tpr[i]);
            roc_curve->add_fpr(fpr[i]);
        }

        auto *record = new Record();
        auto v = record->add_values();
        v->set_id(step);
        v->set_tag(tag);
        v->set_timestamp(walltime);
        v->set_allocated_roc_curve(roc_curve);

        return add_record(record);
    } else {
        throw std::invalid_argument("curve type " + type +
                                    " can not be recognized");
    }
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
