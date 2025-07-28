#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import base64
from typing import List, Optional, Tuple

import kaldi_native_fbank as knf
import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tiny.en"],
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--use-quant",
        type=int,
        default=0,
    )
    return parser.parse_args()


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

    padding = np.zeros((25 * 16000), dtype=np.float32)

    online_whisper_fbank.accept_waveform(16000, wave)
    online_whisper_fbank.accept_waveform(16000, padding)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        features.append(f)

    features = np.stack(features)

    #  if features.shape[0] > 3000:
    #      features = features[:3000]
    #  elif features.shape[0] < 3000:
    #      padding = np.zeros((3000 - features.shape[0], dim), dtype=np.float32)
    #      features = np.concatenate([features, padding], axis=0)

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


class MlxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        print(encoder, decoder)
        self.encoder = mx.import_function(encoder)
        self.decoder = mx.import_function(decoder)
        self.sot = 50257
        self.no_speech = 50361
        self.no_timestamps = 50362
        self.translate = 50357
        self.eot = 50256
        self.blank = 220

    def get_sot_sequence(self):
        # [sot, no_timestamps]
        return [self.sot, self.no_timestamps]

    def get_eot(self):
        return self.eot

    def run_encoder(self, features: mx.array):
        """
        Args:
          features: (1, 3000, 80)
        Returns:
          encoder_out: (1, 15000, n_audio_state)
        """
        mx.eval(features)
        out = self.encoder(features)[0]
        mx.eval(out)
        return out

    def run_decoder(
        self,
        y: mx.array,
        encoder_out: mx.array,
        kv_cache: Optional[List[mx.array]] = None,
    ):
        """
        Args:
          y: (1, 2) or (1, 1) of dtype mx.int32
          encoder_out: Output from self.run_encoder()
          kv_cache: If not None, see ./export.py for what it contains
        Returns:
          logit: (1, 2, vocab_size) or (1, 1, vocab_size)
          out_kv_cache
        """
        print("y", y)
        in_args = [y, encoder_out]
        print("len in args", len(in_args))
        mx.eval(in_args)

        if kv_cache:
            in_args.extend(kv_cache)
            #  print(kv_cache[0].shape, kv_cache[2].shape)
        #  print("in_args", len(in_args))
        print("len in args", len(in_args))

        mx.eval(in_args)
        out = self.decoder(in_args)
        mx.eval(out)
        logits = out[0]
        out_kv_cache = out[1:]
        mx.eval(logits)
        mx.eval(out_kv_cache)
        return logits, out_kv_cache

    def suppress_tokens(self, logits: mx.array, is_initial: bool) -> None:
        print("logits.shape", logits.shape)
        # suppress blank
        if is_initial:
            logits[0, 0, self.eot] = float("-inf")
            logits[0, 0, self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[0, 0, self.no_timestamps] = float("-inf")

        logits[0, 0, self.sot] = float("-inf")
        logits[0, 0, self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[0, 0, self.translate] = float("-inf")
        mx.eval(logits)


def main():
    args = get_args()
    suffix = args.dtype if args.use_quant == 0 else f"{args.dtype}-4bit"
    encoder = f"whisper-{args.model}-encoder.{suffix}.mlxfn"
    decoder = f"whisper-{args.model}-decoder.{suffix}.mlxfn"
    model = MlxModel(encoder=encoder, decoder=decoder)

    features = compute_features("./0.wav", dim=80)
    #  features = compute_features("./1.wav", dim=80)
    features = mx.array(features)
    # features: (1, 3000, 80)

    mx.eval(features)

    encoder_out = model.run_encoder(features)
    mx.eval(encoder_out)
    # encoder_out: (1, 1500, n_audio_state)

    sot_sequence = mx.array([[model.sot]], dtype=mx.int32)
    mx.eval(sot_sequence)

    logits, kv_cache = model.run_decoder(sot_sequence, encoder_out)
    mx.eval(logits)
    mx.eval(kv_cache)

    logits, kv_cache = model.run_decoder(
        mx.array([[model.no_timestamps]], dtype=mx.int32), encoder_out, kv_cache
    )
    mx.eval(logits)
    mx.eval(kv_cache)

    logits = logits[:, -1:]
    mx.eval(logits)
    mx.eval(kv_cache)
    #  model.suppress_tokens(logits, is_initial=True)
    ans = []

    for i in range(200):
        token = logits.squeeze().argmax().item()
        if token == model.get_eot():
            break
        print(
            token,
            logits.sum(),
        )
        print(" kv cache")
        for k in range(4):
            ss = kv_cache[k * 4 : (k + 1) * 4]
            print(
                "  ", k, ss[0].sum(), ss[1].sum(), ss[2].sum(), ss[3].sum(), ss[0].shape
            )
            print(
                "  ",
                k,
                ss[0].mean(),
                ss[1].mean(),
                ss[2].mean(),
                ss[3].mean(),
                ss[0].shape,
            )
        ans.append(token)
        logits, kv_cache = model.run_decoder(
            mx.array([[token]], dtype=mx.int32), encoder_out, kv_cache
        )
        mx.eval(logits)
        mx.eval(kv_cache)
        #  model.suppress_tokens(logits, is_initial=False)
        token = logits.squeeze().argmax().item()
    print(ans)

    token_table = load_tokens(f"whisper-{args.model}-tokens.txt")
    s = b""
    for i in ans:
        if i in token_table:
            s += base64.b64decode(token_table[i])

    print(s.decode().strip())


if __name__ == "__main__":
    main()
