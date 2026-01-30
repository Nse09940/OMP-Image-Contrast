#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <algorithm>
#include <array>

struct Image {
    std::string magic_number;
    int width;
    int height;
    int channels;
    std::vector<uint8_t> pixels;
};

Image read_pnm(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Could not open input file " << filename << std::endl;
        std::exit(1);
    }
    Image img;
    int maxval;
    in >> img.magic_number >> img.width >> img.height >> maxval;
    
    if (img.magic_number == "P5") {
        img.channels = 1;
    } else if (img.magic_number == "P6") {
        img.channels = 3;
    } else {
        std::cerr << "Error: Unsupported PNM format " << img.magic_number << std::endl;
        std::exit(1);
    }

    size_t total_size = static_cast<size_t>(img.width) * img.height * img.channels;
    img.pixels.resize(total_size);
    in.get();
    in.read(reinterpret_cast<char *>(img.pixels.data()), total_size);
    
    return img;
}

void write_pnm(const std::string &filename, const Image &img) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        std::exit(1);
    }
    out << img.magic_number << "\n"
        << img.width << " " << img.height << "\n"
        << "255\n";
    out.write(reinterpret_cast<const char *>(img.pixels.data()), img.pixels.size());
}

void apply_auto_contrast(Image &img, double contrast_coeff, int threads, const std::string &schedule_kind, int chunk_size) {
    int total_pixels = img.width * img.height;
    
    if (schedule_kind == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, chunk_size);
    } else {
        omp_set_schedule(omp_sched_static, chunk_size);
    }

    double t_start = omp_get_wtime();

    std::vector<std::vector<int>> thread_histograms(threads, std::vector<int>(img.channels * 256, 0));

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto &local_hist = thread_histograms[tid];

        #pragma omp for schedule(runtime)
        for (int i = 0; i < total_pixels; ++i) {
            for (int c = 0; c < img.channels; ++c) {
                uint8_t val = img.pixels[i * img.channels + c];
                local_hist[c * 256 + val]++;
            }
        }
    }

    std::vector<int> total_histogram(img.channels * 256, 0);
    for (int c = 0; c < img.channels; ++c) {
        for (int v = 0; v < 256; ++v) {
            for (int t = 0; t < threads; ++t) {
                total_histogram[c * 256 + v] += thread_histograms[t][c * 256 + v];
            }
        }
    }

    int border_pixels = static_cast<int>(contrast_coeff * total_pixels);
    std::vector<int> min_vals(img.channels), max_vals(img.channels);

    for (int c = 0; c < img.channels; ++c) {
        int sum = 0;
        for (int v = 0; v < 256; ++v) {
            sum += total_histogram[c * 256 + v];
            if (sum > border_pixels) {
                min_vals[c] = v;
                break;
            }
        }
        sum = 0;
        for (int v = 255; v >= 0; --v) {
            sum += total_histogram[c * 256 + v];
            if (sum > border_pixels) {
                max_vals[c] = v;
                break;
            }
        }
    }

    int global_min = min_vals[0];
    int global_max = max_vals[0];
    if (img.channels > 1) {
        global_min = std::min({min_vals[0], min_vals[1], min_vals[2]});
        global_max = std::max({max_vals[0], max_vals[1], max_vals[2]});
    }

    std::array<uint8_t, 256> lut;
    if (global_min < global_max) {
        double scale = 255.0 / (global_max - global_min);
        for (int i = 0; i < 256; ++i) {
            int newVal = std::round((i - global_min) * scale);
            lut[i] = static_cast<uint8_t>(std::clamp(newVal, 0, 255));
        }
    } else {
        for (int i = 0; i < 256; ++i) lut[i] = static_cast<uint8_t>(i);
    }

    #pragma omp parallel for schedule(runtime) num_threads(threads)
    for (size_t i = 0; i < img.pixels.size(); ++i) {
        img.pixels[i] = lut[img.pixels[i]];
    }

    double t_end = omp_get_wtime();
    std::printf("Processed in %.2f ms using %d threads.\n", (t_end - t_start) * 1000.0, threads);
}

int main(int argc, char *argv[]) {
    std::string input_file, output_file;
    int threads = 1;
    double contrast_coeff = 0.0;
    std::string schedule = "static";
    int chunk_size = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i < argc - 1) {
            input_file = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i < argc - 1) {
            output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i < argc - 1) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--coeff") == 0 && i < argc - 1) {
            contrast_coeff = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--schedule") == 0 && i < argc - 1) {
            schedule = argv[++i];
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i < argc - 1) {
            chunk_size = std::atoi(argv[++i]);
        }
    }

    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --input <file> --output <file> [--threads <n>] [--coeff <f>] [--schedule <kind>] [--chunk <z>]" << std::endl;
        return 1;
    }

    Image img = read_pnm(input_file);
    apply_auto_contrast(img, contrast_coeff, threads, schedule, chunk_size);
    write_pnm(output_file, img);

    return 0;
}
