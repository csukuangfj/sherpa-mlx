#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import kaldi_native_fbank as knf
import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_whisper import load_models
from mlx_whisper.tokenizer import get_tokenizer


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str, dim: int = 80) -> np.array:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 3000, 80) containing the features.

      Note: In mlx, it expects (1, 3000, 80). In the original whisper, it is
      (1, 80, 3000)
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, wave)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        features.append(f)

    features = np.stack(features)

    log_spec = np.log10(np.maximum(features, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    if mel.shape[0] > 3000:
        mel = mel[:3000]
    elif mel.shape[0] < 3000:
        padding = np.zeros((3000 - mel.shape[0], dim), dtype=np.float32)
        mel = np.concatenate([mel, padding], axis=0)

    return np.ascontiguousarray(mel[None])


"""
tiny.en
ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384,
n_audio_head=6, n_audio_layer=4, n_vocab=51864, n_text_ctx=448,
n_text_state=384, n_text_head=6, n_text_layer=4)
"""


def main():
    m = load_models.load_model("./")
    print(m.is_multilingual, m.num_languages)
    tokenizer = get_tokenizer(m.is_multilingual, num_languages=m.num_languages)
    print("no_speech", tokenizer.no_speech)
    print("no_timestamps", tokenizer.no_timestamps)
    print("translate", tokenizer.translate)
    print("eot", tokenizer.eot)
    print("blank", tokenizer.encode(" ")[0])
    #  return
    print(m.dims)
    sot = mx.array([*tokenizer.sot_sequence, tokenizer.no_timestamps], dtype=mx.int32)
    print(sot)
    print("eot", tokenizer.eot)

    features = compute_features("./0.wav", dim=80)
    #  features = compute_features("./1.wav", dim=80)
    features = mx.array(features)
    # features: (1, 3000, 80)

    encoder_out = m.encoder(features)
    # encoder_out: (1, 1500, 384)

    logits, kv_cache, cross_qk = m.decoder(sot[None], encoder_out)
    logits = logits[:, -1:]
    # logits.shape (1, 1, 51864)
    # len(kv_cache) == 4
    print("logits", logits.shape)  #
    print([(type(v), len(v)) for v in kv_cache])  #
    print([(type(v), type(v[0]), type(v[0][0]), v[0][0].shape) for v in kv_cache])  #
    print(type(cross_qk), len(cross_qk), [v.shape for v in cross_qk])
    print(tokenizer.eot)
    print(type(tokenizer.eot))
    ans = []
    for i in range(100):
        token = logits.squeeze().argmax().item()
        if token == tokenizer.eot:
            break
        print(
            token,
            logits.sum(),
            tokenizer.eot,
            tokenizer.sot_sequence,
        )
        print(" kv cache")
        for k in range(4):
            kv, cross_kv = kv_cache[k]
            print(
                "  ",
                k,
                kv[0].sum(),
                kv[1].sum(),
                cross_kv[0].sum(),
                cross_kv[1].sum(),
                kv[0].shape,
            )
            print(
                "  ",
                k,
                kv[0].mean(),
                kv[1].mean(),
                cross_kv[0].mean(),
                cross_kv[1].mean(),
                kv[0].shape,
            )

        ans.append(token)
        logits, kv_cache, cross_qk = m.decoder(
            mx.array([[token]], dtype=mx.int32), encoder_out, kv_cache
        )
        #  print(kv_cache[0][1][0].shape)
        token = logits.squeeze().argmax().item()
    print(ans)
    print(tokenizer.decode(ans))


if __name__ == "__main__":
    main()
