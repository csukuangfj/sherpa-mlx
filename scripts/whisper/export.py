#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import quantize_model

from mlx_whisper import load_models
from mlx_whisper.tokenizer import get_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
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


def export_encoder(encoder, suffix, feat_dim):
    mx.eval(encoder.parameters())
    x = mx.zeros((1, 3000, feat_dim), dtype=mx.float32)
    mx.eval(x)

    def my_export(x):
        enc_out = encoder(x)
        return enc_out

    mx.export_function(f"whisper-tiny.en-encoder.{suffix}.mlxfn", my_export, x)


def export_decoder(decoder, suffix, dtype, num_layers, dim):
    mx.eval(decoder.parameters())

    enc_out = mx.zeros((1, 1500, dim), dtype=mx.float32)
    sot = mx.array([[50257, 50362]], dtype=mx.int32)  # [sot, no_timestamps]
    x = mx.array([[1793]], dtype=mx.int32)
    mx.eval(enc_out)
    mx.eval(sot)
    mx.eval(x)

    def build_cache(num_tokens):
        in_kv_cache = []
        for i in range(num_layers):
            self_k = mx.zeros((1, num_tokens, dim), dtype=dtype)
            self_v = mx.zeros((1, num_tokens, dim), dtype=dtype)
            cross_k = mx.zeros((1, 1500, dim), dtype=mx.float32)
            cross_v = mx.zeros((1, 1500, dim), dtype=mx.float32)
            in_kv_cache.extend([self_k, self_v, cross_k, cross_v])
        print([t.shape for t in in_kv_cache])
        mx.eval(in_kv_cache)
        return in_kv_cache

    def my_export(*args):
        if len(args) > 2:
            kv_cache = args[2:]

            in_kv_cache = []
            for i in range(num_layers):
                kv = kv_cache[i * 4 : (i + 1) * 4]
                in_kv_cache.append([(kv[0], kv[1]), (kv[2], kv[3])])

            logits, out_kv_cache, _ = decoder(args[0], args[1], in_kv_cache)
        else:
            logits, out_kv_cache, _ = decoder(args[0], args[1])
        # flatten the kv_cache

        cache = []
        for self_kv, cross_kv in out_kv_cache:
            cache.append(self_kv[0])
            cache.append(self_kv[1])
            cache.append(cross_kv[0])
            cache.append(cross_kv[1])

        return logits, *cache

    with mx.exporter(f"whisper-tiny.en-decoder.{suffix}.mlxfn", my_export) as exporter:
        mx.eval(sot)
        mx.eval(x)
        mx.eval(enc_out)

        #  exporter(sot, enc_out)
        exporter(x, enc_out)

        n_text_ctx = 448
        #  n_text_ctx = 300
        for i in range(1, n_text_ctx):
            # max i is 447
            kv_cache = build_cache(i)

            in_args = [x, enc_out, *kv_cache]
            mx.eval(in_args)
            exporter(in_args)


def main():
    args = get_args()
    model = load_models.load_model("./")
    model.eval()
    mx.eval(model.parameters())

    print(model.is_multilingual, model.num_languages)
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    print(model.dims)
    sot = mx.array([*tokenizer.sot_sequence, tokenizer.no_timestamps], dtype=mx.int32)
    mx.eval(sot)

    curr_weights = dict(tree_flatten(model.parameters()))
    if args.dtype == "float32":
        dtype = mx.float32
    elif args.dtype == "float16":
        dtype = mx.float16
    elif args.dtype == "bfloat16":
        dtype = mx.bfloat16
    else:
        assert False, f"Unsupported dtype {args.dtype}"

    suffix = args.dtype if args.use_quant == 0 else f"{args.dtype}-4bit"

    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))
    model.eval()
    mx.eval(model.parameters())

    if args.use_quant:
        model, config = quantize_model(model, {}, q_group_size=64, q_bits=4)
        print("config", config)
    mx.eval(model.parameters())

    mx.eval(model.encoder)
    mx.eval(model.decoder)
    export_encoder(model.encoder, suffix=suffix, feat_dim=80)
    export_decoder(model.decoder, dtype=dtype, suffix=suffix, num_layers=4, dim=384)


if __name__ == "__main__":
    main()
