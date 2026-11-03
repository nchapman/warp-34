# Strata Flash reference preparation

This is explicit operator preparation, not behavior of the portable HTTP client.
The pinned source is `https://github.com/Niko1221/Strata` at
`ffa80d73133ea5c548aefe57889a16e8085a938f`. Build for the actual target GPU:

```sh
export PATH=/opt/cuda/bin:$PATH
cmake -S /tmp/w34-benchmark-refs/Strata \
  -B "$HOME/.cache/w34-benchmark-builds/Strata" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=OFF -DSTRATA_BUILD_DSV41=OFF
cmake --build "$HOME/.cache/w34-benchmark-builds/Strata" --target strata -j 6
uv venv /tmp/w34-benchmark-refs/Strata/.venv --python 3.12
uv pip install --python /tmp/w34-benchmark-refs/Strata/.venv/bin/python \
  -r /tmp/w34-benchmark-refs/Strata/requirements.txt
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 --include 'UD-IQ4_XS/*'
```

The packer needs `gguf-py` from a llama.cpp checkout. This campaign used
`cb7934c52ca8710994b2ecc19775ebefcfdb8d01` at the path below. Set
`STRATA_GGUF_PY` explicitly; the first preparation attempt without it failed,
not a model/runtime failure.

```sh
cd /tmp/w34-benchmark-refs/Strata
STRATA_GGUF_PY=/home/nchapman/Code/warp3090/vendor/llama.cpp/gguf-py \
  .venv/bin/python tools/iq_pack.py \
  --gguf "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" \
  --out "$HOME/.cache/w34-benchmark-models/strata-flash-iq4xs" --compat-bf16
```

Retain `conversions.json`, `compat-bf16.json`, `index.txt` and
`native_experts.txt`. Packing does **not** make this artifact equal to the
AutoRound checkpoint: routed expert formats are mixed, PLE is IQ4_NL, and small
projections are rounded to BF16. Do not add `--experts-bin`, which writes a large
unnecessary expert copy. The HTTP configuration is
[`strata-flash-http.json`](strata-flash-http.json); its absolute paths must be
adapted explicitly on other hosts. The launch command lives in
`flash-next-serving.toml` under `strata-flash-tp1-target`.

The profiles use one GPU, an auto-sized GPU expert cache, a 72-GiB resident budget,
FP16 KV, 8K context and one FIFO sequence. The original **target-only** recipe
failed before readiness: `src/program/generate.cpp:4767` requires `--spec >= 2`
for native IQ packs. Its evidence remains under `results/completion-flash-strata/`.
The separately named `strata-flash-tp1-lookup4` uses
`strata-flash-lookup-http.json`, `--spec 4 --prefill 512`, and no `--mtp`.
Per the pinned `docs/ORCA.md`, absence of `--mtp` means lookup-only drafting;
this is not an MTP comparison and not a target-only run. Request ceilings are strict (`--no-fit-max-tokens`); no silent
context/output shrinkage is allowed. No page-cache flush is claimed. Collect
actual RSS, system memory, GPU memory and startup residency receipts rather than
interpreting the requested budget as measured residency. Build success does not
qualify HTTP behavior or output quality.
