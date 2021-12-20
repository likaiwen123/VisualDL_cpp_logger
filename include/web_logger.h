#ifndef TENSORBOARD_LOGGER_H
#define TENSORBOARD_LOGGER_H

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "crc.h"
#include "event.pb.h"
#include "record.pb.h"

using tensorflow::Event;
using tensorflow::Summary;

using visualdl::Record;

// extract parent dir from path by finding the last slash
std::string get_parent_dir(const std::string &path);

const std::string kProjectorConfigFile = "projector_config.pbtxt";
const std::string kProjectorPluginName = "projector";
const std::string kTextPluginName = "text";

std::string read_binary_file(const std::string &filename);

// todo: limit not checked.
template <typename T>
void calculate_hist_bins(T min, T max, int bins, T &start, T &width) {
    assert(bins > 1);
    assert(max > min);
    T width_min = (max - min) / T(bins);
    T width_max = (max - min) / T(bins - 1);

    double order = floor(log10(width_min));

    width = exp10(order) * ceil(double(width_min) / exp10(order));

    // todo: maybe the number of loop should be limited.
    while (width > width_max) {
        order -= 1;
        width = exp10(order) * ceil(double(width_min) / exp10(order));
    }

    T start_min = max - T(bins) * width;
    T start_max = min;

    int sign = 1;
    if (start_min < 0 && start_max > 0) {
        start = 0.0;
        return;
    } else if (start_min < 0) {
        // both negative, swap and change to positive
        T swap = start_min;
        start_min = -start_max;
        start_max = -swap;
        sign = -1;
    }

    order = floor(log10(start_min));
    start = exp10(order) * ceil(double(start_min) / exp10(order));

    // todo: maybe the number of loop should be limited.
    while (start > start_max) {
        order -= 1;
        start = exp10(order) * ceil(double(start_min) / exp10(order));
    }
    start *= sign;
}

class TensorBoardLogger {
   public:
    explicit TensorBoardLogger(const char *log_file_or_dir,
                               bool visualdl = false,
                               const std::string &suffix = "") {
        bucket_limits_ = nullptr;

        if (visualdl) {
            std::stringstream time_str;
            time_str << std::setw(10) << std::setfill('0') << time(nullptr);
            std::string filename =
                "vdlrecords." + time_str.str() + ".log" + suffix;

            // todo: create when not exists.
            log_dir_ = log_file_or_dir;
            ofs_ = new std::ofstream(
                log_file_or_dir + std::string("/") + filename,
                std::ios::out | std::ios::trunc | std::ios::binary);
        } else {
            ofs_ = new std::ofstream(log_file_or_dir, std::ios::out |
                                                          std::ios::trunc |
                                                          std::ios::binary);
            log_dir_ = get_parent_dir(log_file_or_dir);
        }
        if (!ofs_->is_open())
            throw std::runtime_error("failed to open log_file " +
                                     std::string(log_file_or_dir));
    }
    ~TensorBoardLogger() {
        ofs_->close();
        if (bucket_limits_ != nullptr) {
            delete bucket_limits_;
            bucket_limits_ = nullptr;
        }
    }
    int add_meta(const std::string &tag = std::string("meta_data_tag"),
                 const std::string &display_name = "", int64_t step = 0,
                 time_t timestamp = -1);

    int add_scalar_tb(const std::string &tag, int step, double value);
    int add_scalar(const std::string &tag, int step, double value,
                   time_t walltime = -1);
    int add_scalar_tb(const std::string &tag, int step, float value);

    // https://github.com/dmlc/tensorboard/blob/master/python/tensorboard/summary.py#L127
    template <typename T>
    int add_histogram_tb(const std::string &tag, int step, const T *value,
                         size_t num) {
        if (bucket_limits_ == nullptr) {
            generate_default_buckets();
        }

        std::vector<int> counts(bucket_limits_->size(), 0);
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::lowest();
        double sum = 0.0;
        double sum_squares = 0.0;
        for (size_t i = 0; i < num; ++i) {
            T v = value[i];
            auto lb = std::lower_bound(bucket_limits_->begin(),
                                       bucket_limits_->end(), v);
            counts[lb - bucket_limits_->begin()]++;
            sum += v;
            sum_squares += v * v;
            if (v > max) {
                max = v;
            } else if (v < min) {
                min = v;
            }
        }

        auto *histo = new tensorflow::HistogramProto();
        histo->set_min(min);
        histo->set_max(max);
        histo->set_num(num);
        histo->set_sum(sum);
        histo->set_sum_squares(sum_squares);
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                histo->add_bucket_limit((*bucket_limits_)[i]);
                histo->add_bucket(counts[i]);
            }
        }

        auto *summary = new tensorflow::Summary();
        auto *v = summary->add_value();
        v->set_tag(tag);
        v->set_allocated_histo(histo);

        return add_event(step, summary);
    };

    template <typename T>
    int add_histogram_tb(const std::string &tag, int step,
                         const std::vector<T> &values) {
        return add_histogram_tb(tag, step, values.data(), values.size());
    };

    template <typename T>
    int add_histogram(const std::string &tag, int step, int bins,
                      const T *value, size_t num, time_t walltime = -1) {
        T min = *std::min_element(value, value + num);
        T max = *std::max_element(value, value + num);

        T width, start;
        calculate_hist_bins(min, max, bins, start, width);

        std::vector<T> bin_bounds(bins + 1, 0);
        bin_bounds[0] = start;
        for (size_t t = 1; t < bins + 1; ++t) {
            bin_bounds[t] = start + width * t;
        }

        std::vector<int> count(bins, 0);
        for (size_t i = 0; i < num; ++i) {
            auto ptr = std::lower_bound(bin_bounds.begin(), bin_bounds.end(),
                                        value[i]);
            if (ptr == bin_bounds.end()) {
                count[bins - 1]++;
            } else {
                count[ptr - bin_bounds.begin()]++;
            }
        }

        auto *hist = new visualdl::Record_Histogram();
        for (size_t i = 0; i < bins + 1; ++i) {
            hist->add_bin_edges(bin_bounds[i]);
        }
        for (size_t i = 0; i < bins; ++i) {
            hist->add_hist(count[i]);
        }

        auto *record = new visualdl::Record();
        auto v = record->add_values();
        v->set_id(step);
        v->set_tag(tag);
        v->set_timestamp(walltime);
        v->set_allocated_histogram(hist);

        return add_record(record);
    };

    template <typename T>
    int add_histogram(const std::string &tag, int step, int bins,
                      const std::vector<T> &values, time_t walltime = -1) {
        return add_histogram(tag, step, bins, values.data(), values.size(),
                             walltime);
    };

    // metadata (such as display_name, description) of the same tag will be
    // stripped to keep only the first one.
    int add_image_tb(const std::string &tag, int step,
                     const std::string &encoded_image, int height, int width,
                     int channel, const std::string &display_name = "",
                     const std::string &description = "");
    int add_image(const std::string &tag, int step,
                  const std::string &encoded_image, time_t walltime = -1);
    int add_image_from_path(const std::string &tag, int step,
                            const std::string &path, time_t walltime = -1);
    int add_images_tb(const std::string &tag, int step,
                      const std::vector<std::string> &encoded_images,
                      int height, int width,
                      const std::string &display_name = "",
                      const std::string &description = "");
    int add_audio_tb(const std::string &tag, int step,
                     const std::string &encoded_audio, float sample_rate,
                     int num_channels, int length_frame,
                     const std::string &content_type,
                     const std::string &display_name = "",
                     const std::string &description = "");
    int add_audio(const std::string &tag, int step,
                  const std::string &encoded_audio, float sample_rate,
                  time_t walltime = -1);
    int add_audio_from_path(const std::string &tag, int step,
                            const std::string &path, float sample_rate,
                            time_t walltime = -1);

    int add_text_tb(const std::string &tag, int step, const char *text);
    int add_text(const std::string &tag, int step, const std::string &text,
                 time_t walltime = -1);

    // `tensordata` and `metadata` should be in tsv format, and should be
    // manually created before calling `add_embedding_tb`
    //
    // `tensor_name` is mandated to differentiate tensors
    //
    // TODO add sprite image support
    int add_embedding_tb(
        const std::string &tensor_name, const std::string &tensordata_path,
        const std::string &metadata_path = "",
        const std::vector<uint32_t> &tensor_shape = std::vector<uint32_t>(),
        int step = 1 /* no effect */);
    // write tensor to binary file
    int add_embedding_tb(
        const std::string &tensor_name,
        const std::vector<std::vector<float>> &tensor,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);
    int add_embedding_tb(
        const std::string &tensor_name, const float *tensor,
        const std::vector<uint32_t> &tensor_shape,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);

    int add_embeddings(const std::string &tag,
                       const std::vector<std::vector<float>> &mat,
                       const std::vector<std::vector<std::string>> &metadata,
                       const std::vector<std::string> &metadata_header =
                           std::vector<std::string>(),
                       time_t walltime = -1);
    int add_embeddings(const std::string &tag,
                       const std::vector<std::vector<float>> &mat,
                       const std::vector<std::string> &metadata,
                       const std::vector<std::string> &metadata_header =
                           std::vector<std::string>(),
                       time_t walltime = -1);

   private:
    int generate_default_buckets();
    int add_event(int64_t step, Summary *summary);
    inline int add_record(Record *record) { return write(*record); }

    int write(Event &event);
    int write(Record &record);

    std::string log_dir_;
    std::ofstream *ofs_;
    std::vector<double> *bucket_limits_;
};  // class TensorBoardLogger

#endif  // TENSORBOARD_LOGGER_H
