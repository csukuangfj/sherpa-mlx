#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import kaldi_native_fbank as knf
import librosa
import numpy as np
import torch
import yaml

from torch_model import Paraformer


def load_cmvn():
    neg_mean = None
    inv_std = None

    with open("am.mvn") as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]
            t = list(map(lambda x: float(x), t))

            if neg_mean is None:
                neg_mean = np.array(t, dtype=np.float32)
            else:
                inv_std = np.array(t, dtype=np.float32)

    return neg_mean, inv_std


def compute_feat(filename):
    sample_rate = 16000
    samples, _ = librosa.load(filename, sr=sample_rate)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype
    print("features sum", features.sum(), features.size)

    window_size = 7  # lfr_m
    window_shift = 6  # lfr_n

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    neg_mean, inv_std = load_cmvn()
    features = (features + neg_mean) * inv_std
    return features


def load_model():

    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    m = Paraformer(
        input_size=560,
        vocab_size=8404,
        encoder_conf=config["encoder_conf"],
        decoder_conf=config["decoder_conf"],
        predictor_conf=config["predictor_conf"],
    )
    m.eval()

    state_dict = torch.load("./model_state_dict.pt", map_location="cpu")["state_dict"]
    m.load_state_dict(state_dict)
    del state_dict

    return m


def load_tokens():
    ans = dict()
    i = 0
    with open("tokens.txt", encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


@torch.no_grad()
def main():
    features = compute_feat("./16.wav")
    features = torch.from_numpy(features).unsqueeze(0)
    print("features.shape", features.shape)
    model = load_model()
    print("computing")
    out = model(features)
    if len(out) == 0:
        print("No speech in the audio file")
        return
    decoder_out, pre_token_length = out

    yseq = decoder_out[0].argmax(dim=-1).tolist()
    tokens = load_tokens()
    words = [tokens[i] for i in yseq if i not in (1, 2)]
    print(words)
    text = "".join(words)
    print(text)


if __name__ == "__main__":
    main()
