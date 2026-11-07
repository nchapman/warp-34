# Server recipes for short tooling checks

Use the same native `lm-eval.yaml` and `--limit 10` client command in the
[README](README.md) for every server. Launch one server at a time on the pod,
wait for `/v1/models` to return `warp34-quality`, then run both tasks. Do not
infer readiness solely from one server's log wording. Bind to localhost.

These are integration checks, **not a matched quality or performance comparison**:
warp-34 uses HGPTQ/TQ4; the donor checks use regular GPTQ/native KV. MTP uses
built-in target weights, not a separately chosen DFlash2 model. MTP flag syntax
and implementation differ by server.

## Environment and checkpoint

Keep server dependencies separate from the lm-eval environment and from one
another. The pod recipes use Python 3.12 and these server releases:

```sh
uv venv --python 3.12 /workspace/vllm-venv
uv pip install --python /workspace/vllm-venv/bin/python 'vllm==0.31.0'
uv venv --python 3.12 /workspace/sglang-venv
uv pip install --python /workspace/sglang-venv/bin/python 'sglang[all]==0.5.19'
```

These pin server releases, not all server dependencies. Preserve `uv pip freeze`
for each server in the run receipt. The evaluator has its own hashed dependency
lock. No Docker-in-Docker or driver changes are required.

For the donor commands below, `MODEL` is an absolute local snapshot path for
`nchapman/Qwen3.8-27B-W4A16-GPTQ` revision
`0df81eadcfd59601cb82ccece2f16ea7d39244bf`. This is **not** the HGPTQ checkpoint
used by the initial warp-34 checks. The [README](README.md) gives warp-34's
HGPTQ/MTP-3 command. Retain exact snapshot identities even though the API alias
is shared.

```sh
export PATH="/usr/local/cuda/bin:$PATH"
export HF_HOME=/workspace/.hf_home
export MODEL="$HF_HOME/hub/models--nchapman--Qwen3.8-27B-W4A16-GPTQ/snapshots/0df81eadcfd59601cb82ccece2f16ea7d39244bf"
```

## vLLM 0.31.0 — 27B TP2/MTP-3

```sh
CUDA_VISIBLE_DEVICES=0,1 /workspace/vllm-venv/bin/vllm serve "$MODEL" \
  --served-model-name warp34-quality --dtype half --tensor-parallel-size 2 \
  --language-model-only --max-model-len 8192 --max-num-seqs 8 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float32 --generation-config vllm --seed 34 \
  --cudagraph-capture-sizes 1 2 4 8 --host 127.0.0.1 --port 18034 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The short check completed with 20 successful requests. Initial compilation and
model loading are separate from the small evaluation workload.

## SGLang 0.5.19 — 27B TP2/native MTP probe

This recipe passed the 20-request tooling check; that does not qualify other
GPTQ/MTP combinations. SGLang exposes this native MTP path via EAGLE flags and
the same target snapshot as the draft path:

```sh
CUDA_VISIBLE_DEVICES=0,1 SGLANG_MAMBA_CONV_DTYPE=float16 \
  /workspace/sglang-venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name warp34-quality \
  --dtype float16 --tp-size 2 --context-length 8192 \
  --mamba-ssm-dtype float32 --mem-fraction-static 0.88 \
  --chunked-prefill-size 1024 --max-running-requests 8 --max-mamba-cache-size 32 \
  --cuda-graph-bs-decode 1 2 4 8 --cuda-graph-bs-prefill 512 1024 \
  --reasoning-parser qwen3 --enable-cache-report --enable-metrics \
  --random-seed 34 --host 127.0.0.1 --port 18034 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path "$MODEL"
```

`SGLANG_MAMBA_CONV_DTYPE=float16` keeps convolution-cache dtype consistent with
the FP16 target; the SSM state remains FP32. Do not copy warp-34's TQ4 flags or
vLLM's `--language-model-only` into this command.

## Run the client

From the repository root, after model discovery succeeds:

```sh
curl --fail http://127.0.0.1:18034/v1/models
export NLTK_DATA="$HOME/nltk_data"
OUT="benchmarks/quality/results/$(date -u +%Y%m%dT%H%M%SZ)-server-model-smoke10"
test ! -e "$OUT" && mkdir -p "$OUT" || exit 1
benchmarks/quality/.venv/bin/lm-eval run \
  --config benchmarks/quality/lm-eval.yaml --limit 10 \
  --output_path "$OUT/upstream" > "$OUT/lm-eval.log" 2>&1
```

Replace the output label with the actual engine/model. Preserve provenance and
server logs as described in the README. Stop the server before starting the
next deployment. Token-budget finishes are valid upstream-scored outputs, not
items to remove; serving failures and empty responses require investigation.
