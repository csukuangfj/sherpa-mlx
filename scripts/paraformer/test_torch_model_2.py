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


def get_acoustic_embedding(alpha: torch.Tensor, hidden: torch.Tensor):
    """
    Args:
      alpha: (T,)
      hidden: (T, C)
    Returns:
      acoustic_embeds: (num_tokens, C)
    """
    print(alpha.shape, hidden.shape)
    alpha = alpha.tolist()
    acc = 0
    num_tokens = 0

    embeddings = []
    cur_embedding = torch.zeros(hidden.shape[1], dtype=torch.float32)

    for i, w in enumerate(alpha):
        acc += w
        if acc >= 1:
            overflow = acc - 1
            remain = w - overflow
            cur_embedding += remain * hidden[i]
            embeddings.append(cur_embedding)

            cur_embedding = overflow * hidden[i]
            acc = overflow
        else:
            cur_embedding += w * hidden[i]

    if len(embeddings) == 0:
        raise ValueError("No speech in the audio file")

    embeddings = torch.stack(embeddings)
    return embeddings


@torch.no_grad()
def main():
    features = compute_feat("./1.wav")
    features = torch.from_numpy(features).unsqueeze(0)
    print("features.shape", features.shape)
    model = load_model()
    print("computing")

    encoder_out = model.encoder(features)
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = model.predictor(
        encoder_out
    )

    print("pre_acoustic_embeds.shape", pre_acoustic_embeds.shape)
    print("pre_acoustic_embeds.sum", pre_acoustic_embeds[0, :30].sum())

    pre_acoustic_embeds = get_acoustic_embedding(
        alphas[0, :-1], encoder_out[0]
    ).unsqueeze(0)

    print("pre_acoustic_embeds.shape", pre_acoustic_embeds.shape)
    print("pre_acoustic_embeds.sum", pre_acoustic_embeds[0, :30].sum())

    decoder_out = model.decoder(encoder_out, pre_acoustic_embeds)
    print("decoder_out.shape", decoder_out.shape)

    yseq = decoder_out[0].argmax(dim=-1).tolist()
    print(yseq)
    tokens = load_tokens()
    words = [tokens[i] for i in yseq if i not in (1, 2)]
    print(words)
    text = "".join(words)
    print(text)


if __name__ == "__main__":
    main()
