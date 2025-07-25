// sherpa-mlx/csrc/voice-activity-detector.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation

#ifndef SHERPA_MLX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_MLX_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#include "sherpa-mlx/csrc/vad-model-config.h"

namespace sherpa_mlx {

struct SpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
};

// this class is not thread-safe
class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

  template <typename Manager>
  VoiceActivityDetector(Manager *mgr, const VadModelConfig &config,
                        float buffer_size_in_seconds = 60);

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  bool Empty() const;
  void Pop();
  void Clear();

  // It is an error to call Front() if Empty() returns true.
  //
  // The returned reference is valid until the next call to any
  // methods of VoiceActivityDetector.
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  // It is empty if IsSpeechDetected() returns false
  SpeechSegment CurrentSpeechSegment() const;

  void Reset();

  // At the end of the utterance, you can invoke this method so that
  // the last speech segment can be detected.
  void Flush();

  const VadModelConfig &GetConfig() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mlx

#endif  // SHERPA_MLX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
