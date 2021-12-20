#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "web_logger.h"

using namespace std;

int test_log_scalar(TensorBoardLogger& logger) {
    cout << "test log scalar" << endl;
    default_random_engine generator;
    normal_distribution<double> default_distribution(0, 1.0);
    for (int i = 0; i < 10; ++i) {
        logger.add_scalar_tb("scalar", i, default_distribution(generator));
    }

    return 0;
}

int test_log_histogram(TensorBoardLogger& logger) {
    cout << "test log histogram" << endl;
    default_random_engine generator;
    for (int i = 0; i < 10; ++i) {
        normal_distribution<double> distribution(i * 0.1, 1.0);
        vector<float> values;
        for (int j = 0; j < 10000; ++j)
            values.push_back(distribution(generator));
        logger.add_histogram_tb("histogram", i, values);
    }

    return 0;
}

int test_log_image(TensorBoardLogger& logger) {
    cout << "test log image" << endl;
    // read images
    auto image1 = read_binary_file("./assets/text.png");
    auto image2 = read_binary_file("./assets/audio.png");

    // add single image
    logger.add_image_tb("TensorBoard Text Plugin", 1, image1, 1864, 822, 3,
                        "TensorBoard", "Text");
    logger.add_image_tb("TensorBoard Audo Plugin", 1, image2, 1766, 814, 3,
                        "TensorBoard", "Audio");

    // add multiple images
    // FIXME This seems doesn't work anymore.
    // logger.add_images_tb(
    //     "Multiple Images", 1, {image1, image2}, 1502, 632, "test", "not
    //     working");

    return 0;
}

int test_log_audio(TensorBoardLogger& logger) {
    cout << "test log audio" << endl;
    auto audio = read_binary_file("./assets/file_example_WAV_1MG.wav");
    logger.add_audio_tb(
        "Audio Sample", 1, audio, 8000, 2, 8000 * 16 * 2 * 33, "audio/wav",
        "Impact Moderato",
        "https://file-examples.com/index.php/sample-audio-files/"
        "sample-wav-download/");

    return 0;
}

int test_log_text(TensorBoardLogger& logger) {
    cout << "test log text" << endl;
    logger.add_text_tb("Text Sample", 1, "Hello World");

    return 0;
}

int test_log_embedding(TensorBoardLogger& logger) {
    cout << "test log embedding" << endl;
    // test add embedding
    logger.add_embedding_tb("vocab", "../assets/vecs.tsv",
                            "../assets/meta.tsv");
    logger.add_embedding_tb("another vocab without labels",
                            "../assets/vecs.tsv");

    // test add binary embedding
    vector<vector<float>> tensor;
    string line;
    ifstream vec_file("assets/vecs.tsv");
    uint32_t num_elements = 1;
    while (getline(vec_file, line)) {
        istringstream values(line);
        vector<float> vec;
        copy(istream_iterator<float>(values), istream_iterator<float>(),
             back_inserter(vec));
        num_elements += vec.size();
        tensor.push_back(vec);
    }
    vec_file.close();

    vector<string> meta;
    ifstream meta_file("assets/meta.tsv");
    while (getline(meta_file, line)) {
        meta.push_back(line);
    }
    meta_file.close();
    logger.add_embedding_tb("binary tensor", tensor, "tensor.bin", meta,
                            "binary_tensor.tsv");

    // test tensor stored as 1d array
    float* tensor_1d = new float[num_elements];
    for (size_t i = 0; i < tensor.size(); i++) {
        const auto& vec = tensor[i];
        memcpy(tensor_1d + i * vec.size(), vec.data(),
               vec.size() * sizeof(float));
    }
    vector<uint32_t> tensor_shape;
    tensor_shape.push_back(tensor.size());
    tensor_shape.push_back(tensor[0].size());
    logger.add_embedding_tb("binary tensor 1d", tensor_1d, tensor_shape,
                            "tensor_1d.bin", meta, "binary_tensor_1d.tsv");
    delete[] tensor_1d;

    return 0;
}

int test_log(const char* log_file) {
    TensorBoardLogger logger(log_file);

    test_log_scalar(logger);
    //    test_log_histogram(logger);
    //    test_log_image(logger);
    //    test_log_audio(logger);
    //    test_log_text(logger);
    //    test_log_embedding(logger);

    return 0;
}

int test_log_vdl_scalar(TensorBoardLogger& logger,
                        default_random_engine& generator,
                        normal_distribution<double>& default_distribution) {
    cout << "test vdl log scalar" << endl;
    for (int i = 0; i < 10; ++i) {
        logger.add_scalar("scalar_vdl", i, default_distribution(generator));
    }
    return 0;
}

int test_log_vdl_image(TensorBoardLogger& logger) {
    // todo: MatLab figure, image matrix not supported
    cout << "test vdl log image" << endl;
    logger.add_image("dog", 1, read_binary_file("./dog.jpg"));
    logger.add_image("dog", 2, read_binary_file("./2.jpg"));
    logger.add_image("dog", 3, read_binary_file("./dynamic_display.gif"));

    logger.add_image_from_path("gif", 10, "./dynamic_display.gif");

    return 0;
}

int test_log_vdl_audio(TensorBoardLogger& logger) {
    // todo: check more audio file types.
    cout << "test vdl log audio" << endl;
    logger.add_audio("audio", 1, read_binary_file("./example.wav"), 8000);
    logger.add_audio_from_path("audio", 2, "./testing.wav", 8000);

    return 0;
}

int test_log_vdl_text(TensorBoardLogger& logger) {
    cout << "test vdl log text" << endl;
    logger.add_text("text vdl", 5, "hello");
    logger.add_text("text vdl", 10, "world");
    logger.add_text("vdl", 10, "hello world");

    return 0;
}

int test_log_vdl_histogram(TensorBoardLogger& logger,
                           default_random_engine& generator) {
    // note: maybe different from the plot from python api, as the bin bounds
    //       settings are calculated differently.
    cout << "test vdl log histogram" << endl;
    for (int i = 0; i < 3; ++i) {
        normal_distribution<double> distribution(i * 1, 1.0);
        vector<float> values(10000, 0.0);
        for (int j = 0; j < 10000; ++j) {
            values[j] = distribution(generator);
        }
        logger.add_histogram("hist_" + to_string(i), 0, 10, values);
    }

    return 0;
}

int test_log_vdl_embeddings(TensorBoardLogger& logger) {
    // todo: path reading not supported.
    cout << "test vdl log embeddings" << endl;
    vector<vector<float>> embs{
        {1.3561076367500755, 1.3116267195134017, 1.6785401875616097},
        {1.1039614644440658, 1.8891609992484688, 1.32030488587171},
        {1.9924524852447711, 1.9358920727142739, 1.2124401279391606},
        {1.4129542689796446, 1.7372166387197474, 1.7317806077076527},
        {1.3913371800587777, 1.4684674577930312, 1.521413635247637}};
    vector<string> metadata = {"label_1", "label_2", "label_3", "label_4",
                               "label_5"};
    vector<string> metaheader;

    // todo: failed to load data
    // vector<string> metaheader = {"label_1"};
    logger.add_embeddings("single embs", embs, metadata, metaheader, 0);

    vector<vector<string>> labels{
        {"label_a_1", "label_a_2", "label_a_3", "label_a_4", "label_a_5"},
        {"label_b_1", "label_b_2", "label_b_3", "label_b_4", "label_b_5"}};
    vector<string> label_meta{"label_a", "label_b"};
    logger.add_embeddings("embs", embs, labels, label_meta);

    return 0;
}

int test_log_vdl_hparams(const string& dir1, const string& dir2) {
    cout << "test vdl log hparams" << endl;

    TensorBoardLogger logger1(dir1.c_str(), true);
    TensorBoardLogger logger2(dir2.c_str(), true);

    logger1.add_hparams({{"lr", "0.1"}, {"bsize", "1"}, {"opt", "sgd"}},
                        {"hparam/accuracy", "hparam/loss"});
    for (size_t i = 0; i < 10; ++i) {
        logger1.add_scalar("hparam/accuracy", i, 1.0 * i);
        logger1.add_scalar("hparam/loss", i, 2.0 * i);
    }

    logger2.add_hparams({{"lr", "0.2"}, {"bsize", "2"}, {"opt", "relu"}},
                        {"hparam/accuracy", "hparam/loss"});
    for (size_t i = 0; i < 10; ++i) {
        logger2.add_scalar("hparam/accuracy", i, 1.0 / double(i + 1));
        logger2.add_scalar("hparam/loss", i, 5.0 * i);
    }

    return 0;
}

int test_log_vdl(TensorBoardLogger& logger) {
    // todo:
    //       hparams, pr_curve, roc_curve
    default_random_engine generator;
    normal_distribution<double> default_distribution(0, 1.0);

    test_log_vdl_scalar(logger, generator, default_distribution);
    test_log_vdl_image(logger);
    test_log_vdl_audio(logger);
    test_log_vdl_text(logger);
    test_log_vdl_histogram(logger, generator);
    test_log_vdl_embeddings(logger);

    return 0;
}

int test_vdl(const char* log_dir) {
    TensorBoardLogger logger(log_dir, true);
    logger.add_meta();
    test_log_vdl(logger);
    return 0;
}

int main(int argc, char* argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    int ret = test_log("./demo/tfevents.pb");
    assert(ret == 0);

    ret = test_vdl("./logs/out");
    assert(ret == 0);

    ret = test_log_vdl_hparams("./logs/hparam/1", "./logs/hparam/2");
    assert(ret == 0);

    // Optional:  Delete all global objects allocated by libprotobuf.
    // google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
