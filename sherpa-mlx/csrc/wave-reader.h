// sherpa-mlx/csrc/wave-reader.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation

#ifndef SHERPA_MLX_CSRC_WAVE_READER_H_
#define SHERPA_MLX_CSRC_WAVE_READER_H_

#include <istream>
#include <string>
#include <vector>

namespace sherpa_mlx {

/** Read a wave file with expected sample rate.

    @param filename Path to a wave file. Supports multi-channel WAV files
                    with 8, 16, or 32-bit PCM (integer or floating-point)
                    encoding. Only the first channel is returned.
    @param sampling_rate  On return, it contains the sampling rate of the file.
    @param is_ok On return it is true if the reading succeeded; false otherwise.

    @return Return wave samples normalized to the range [-1, 1).
 */
std::vector<float> ReadWave(const std::string &filename, int32_t *sampling_rate,
                            bool *is_ok);

std::vector<float> ReadWave(std::istream &is, int32_t *sampling_rate,
                            bool *is_ok);

std::vector<std::vector<float>> ReadWaveMultiChannel(std::istream &is,
                                                     int32_t *sampling_rate,
                                                     bool *is_ok);

std::vector<std::vector<float>> ReadWaveMultiChannel(
    const std::string &filename, int32_t *sampling_rate, bool *is_ok);

}  // namespace sherpa_mlx

#endif  // SHERPA_MLX_CSRC_WAVE_READER_H_
