# Quality regression evaluations — upstream lm-eval

Run **EleutherAI lm-evaluation-harness directly** against warp-34's chat API.
No custom tasks, prompt rewrites, scoring, answer extraction, subset selection, model
adapter, aggregate quality score, or evaluator fork. `lm-eval.yaml` is a native
upstream **run configuration**, not a task definition. The tests here check that
integration configuration; they do not implement or change any benchmark.

The immediate goal is **tooling qualification across servers and models**:
verify chat requests, scoring and retained outputs with **10 examples per task**.
These tiny runs are not quality baselines or rankings. Once the workflow works
across deployments, larger evaluations can support specific quality comparisons.
Quality and performance runs are separate; do not run them concurrently on the
same server/GPU. Full evaluations require an explicit decision, not this setup pass.

Use the [automated FP16-KV suite](AUTOMATION.md) to run warp-34,
llama.cpp/Unsloth, vLLM and SGLang sequentially with one command. The checked-in
runner manages server lifecycle and invokes the unchanged upstream CLI; it is
not an evaluator adapter. Earlier [mixed-KV single-GPU checks](single-gpu.md),
[TP2 qualification](QUALIFICATION.md) and [TP2 recipes](servers.md) remain
historical records, not FP16-suite results.

## Exact evaluator and tasks

- `lm-eval[api,ifeval]==0.4.13`, upstream tag `v0.4.13`, source commit
  `ddd67220430a2470529f25fd5c05a576ca1057a0`.
- `requirements.lock` pins transitive Python dependencies and package hashes for
  the Python 3.12 environment. No CUDA/PyTorch model backend is needed by this
  HTTP client. The tested environment is Linux/Python 3.12.
- **`ifeval`**: full `google/IFEval` train split (541 evaluation examples), zero
  few-shot examples, task version 4.0, greedy generation, 1,280 output tokens.
  Report all four upstream prompt/instruction strict/loose accuracy metrics.
- **`gsm8k`** (not `gsm8k_cot` or another variant): full `openai/gsm8k` main/test
  split (1,319 examples), five upstream training examples as few-shot context,
  task version 3.0. Report both `exact_match,strict-match` and
  `exact_match,flexible-extract`, not a selectively chosen extraction score.
  Its YAML does not set an output budget; the pinned API backend defaults to
  **256 output tokens**. We deliberately do not override it.

Upstream task sampling settings, few-shot selection, seed defaults, stop strings,
filters and scoring are untouched. Few-shot chat formatting follows the pinned
release's default (`fewshot_as_multiturn=true`). There is no response cache,
`--limit`, `--samples`, `--num_fewshot`, or custom system prompt in our config.
The qualification commands below explicitly apply native `--limit 10`.

Dataset and NLTK resources are downloaded by their standard loaders; the Python
lock **does not pin those external bytes**. Retain resolved dataset cache refs,
logged samples, dataset fingerprints where available and the NLTK resource hash
with each run. Do not assume two runs evaluated identical data merely because
the task names match. No task YAML is patched to force a different dataset.

## Install an isolated environment

From the repository root (requires `uv` and Python 3.12):

```sh
uv venv --python 3.12 benchmarks/quality/.venv
uv pip sync --python benchmarks/quality/.venv/bin/python \
  --require-hashes benchmarks/quality/requirements.lock

# NLTK requires a private location; /workspace may be group/world-writable.
export NLTK_DATA="$HOME/nltk_data"
mkdir -p "$NLTK_DATA"
chmod 700 "$NLTK_DATA"
benchmarks/quality/.venv/bin/python -m nltk.downloader -d "$NLTK_DATA" punkt_tab

mise run quality-check
benchmarks/quality/.venv/bin/lm-eval validate --tasks ifeval,gsm8k
```

`quality-check` is an opt-in CPU-only configuration check; it is not added to the
normal project check's dependency/install footprint. In v0.4.13, `lm-eval
validate` checks task discovery, not an actual server run or full scoring.

To intentionally refresh dependencies, regenerate the lock, review the diff,
and establish a new evaluator baseline rather than comparing across versions:

```sh
uv pip compile benchmarks/quality/requirements.in --python-version 3.12 \
  --generate-hashes --output-file benchmarks/quality/requirements.lock
```

## Start a server, then invoke the upstream CLI

Launch exactly one server in a separate terminal. The primary user-facing setup uses
**Qwen3.8-27B HGPTQ, MTP-3, one GPU and FP16 KV**, with a 4K context. MTP-3 is the common quality-run
configuration across our four model families; it uses the checkpoint's built-in
MTP weights rather than a separately selected DFlash2 drafter.

```sh
export PATH="/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
# Initial pin: nchapman/Qwen3.8-27B-W4A16-HGPTQ
# revision 1de743eedefdb7422e6b171174c7080204f5b005
export MODEL=/absolute/path/to/pinned/model/snapshot
CUDA_VISIBLE_DEVICES=0 ./target/release/w34 serve \
  --model "$MODEL" --offline --tp 1 \
  --kv-format fp16 --kv-pages 256 --max-num-seqs 4 \
  --speculative mtp --mtp-k 3 \
  --served-model-name warp34-quality --host 127.0.0.1 --port 18034
```

Use the same alias for other checkpoints but record their actual identities in
the run receipt—`warp34-quality` is not provenance. Change TP/KV/speculation only
as an explicitly labeled experiment. The four model families remain 9B, 27B,
35B-A3B and Flash-Next; a first 27B result does not qualify the other three.
The earlier Flash-Next/plain run was interrupted at the user's request before
scoring and is not a completed evaluation. Restart full tasks when changing
model or decoding mode; never combine partial outputs across deployments.

In the client terminal, from the repository root:

```sh
export NLTK_DATA="$HOME/nltk_data"
OUT="benchmarks/quality/results/$(date -u +%Y%m%dT%H%M%SZ)-27b-single-mtp3-smoke10"
# Never reuse an existing run directory.
test ! -e "$OUT" && mkdir -p "$OUT" || exit 1
cp benchmarks/quality/lm-eval.yaml "$OUT/"
cp benchmarks/quality/requirements.in benchmarks/quality/requirements.lock "$OUT/"
uv pip freeze --python benchmarks/quality/.venv/bin/python > "$OUT/packages.txt"
sha256sum "$NLTK_DATA/tokenizers/punkt_tab.zip" > "$OUT/nltk-resource.sha256"

benchmarks/quality/.venv/bin/lm-eval run \
  --config benchmarks/quality/lm-eval.yaml --limit 10 \
  --output_path "$OUT/upstream" > "$OUT/lm-eval.log" 2>&1
```

This runs 10 examples from each task (20 total). For separate upstream result bundles, invoke the same
command with `--tasks ifeval` and `--tasks gsm8k`, each with its own output path.
For an already running server, these direct commands are sufficient. The
checked-in automated runner invokes the same two native task commands and adds
server supervision, failure handling and retained evidence.

## Abbreviated smoke runs

Use upstream lm-eval's native `--limit 50` to evaluate **50 examples per task**
(100 total), leaving the shared configuration and upstream scoring unchanged:

```sh
OUT="benchmarks/quality/results/$(date -u +%Y%m%dT%H%M%SZ)-27b-single-mtp3-smoke50"
test ! -e "$OUT" && mkdir -p "$OUT" || exit 1
benchmarks/quality/.venv/bin/lm-eval run \
  --config benchmarks/quality/lm-eval.yaml --limit 50 \
  --output_path "$OUT/upstream" > "$OUT/lm-eval.log" 2>&1
```

Use the same environment, server setup and provenance receipts as the full run
above. The native results record the limit, and `log_samples` still retains
individual examples. Add `--tasks ifeval` or `--tasks gsm8k` to smoke-test only
one task. For a very quick plumbing check, use `--limit 10` instead.

- **Smoke10:** current tooling qualification mode across servers and models.
- **Smoke50:** optional larger check for gross quality failures.
- **Full:** omit `--limit`; required for reported benchmark scores and closer
  quality comparisons. The checked-in YAML continues to default to full runs.

The limit selects the upstream evaluation prefix, not a random or stratified
sample. Five-shot GSM8K context and task generation settings are unchanged.
Compare accepted and candidate builds with the **same limit**, data and serving
settings. Label abbreviated results as smoke results, not full IFEval/GSM8K
scores; do not mix them with full results or use them to qualify a release.
With only 50 examples, one GSM8K answer changes accuracy by two percentage
points, and the prefix can miss whole failure categories. Similar smoke scores
are a reason to proceed to a full run, not evidence of close quality parity.

Do not run a smoke test concurrently with another evaluation on the same server.

### Chat integration — no double templating

`local-chat-completions`, `tokenizer_backend: null`, and
`apply_chat_template: true` make upstream lm-eval send a list of chat messages;
**the server renders the checkpoint's chat template exactly once**. We do not
render a tokenizer template client-side and embed it in another user message.

The one model-specific generation setting is
`chat_template_kwargs: {enable_thinking: false}`. It selects Qwen non-thinking
mode without injecting instructions into task prompts or stripping answers.
There is no custom reasoning filter or `think_end_token`. Name this setting in
comparisons; these are not thinking-enabled scores. The pinned upstream adapter
forwards this parameter while preserving the tasks' generation settings.

The client is sequential (`num_concurrent=1`), with a 1,800-second HTTP timeout.
The server example has a 4K total context pool. `max_length=4096` records the
intended client configuration, but with no client tokenizer it is **not** proof
that inputs fit or that tokens were counted. Preserve any capacity failure;
do not silently trim prompts, lower output budgets or discard examples.

## Evidence and interpreting regressions

Keep the **original upstream results JSON and sample JSONL** produced by
`--log_samples`, plus the complete evaluator log. Do not rewrite scores or drop
failed items. Retain alongside them:

- Engine source commit, binary SHA-256 and exact server command.
- Target/drafter revisions and artifact inventories; tokenizer/template identity.
- GPU/CPU/topology, KV format, TP, speculation and thinking settings.
- The run YAML, package lock/freeze, dataset cache refs and NLTK resource hash.
- Server startup log, model discovery and server error/request counters.

Check native `n-samples`: original counts should be 541/1,319 and effective
counts should match the requested limit (10 for qualification). Inspect unique
sample document IDs, process exit, server failures and evaluator warnings.
GSM8K logs one sample row per filter: its two standard filters yield 20 JSONL
rows for 10 distinct evaluated documents; that is not 20 API requests. The upstream API adapter does not log
all raw HTTP response metadata in samples; do not claim finish-reason or
truncation coverage that was not captured. It can also turn some malformed
responses into empty strings, so an exit code alone is insufficient evidence
of a healthy serving run. Empty/truncated outputs remain scored by upstream;
never exclude them to improve a result.

Compare **unmodified upstream metrics** for accepted and candidate builds using
identical evaluator/data/checkpoint/settings. Inspect logged examples when
scores move. There is no home-grown pass threshold or composite quality score.
A new quantization/KV policy is a separate experiment, not an engine-only
regression comparison. Lossless speculative parity tests remain separate too.

This chat backend cannot run standard log-likelihood/MCQ tasks such as MMLU.
Do not substitute prompted-letter generation and call it standard MMLU.

## Upstream references

- [v0.4.13 release](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13)
- [API guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/docs/API_guide.md)
- [IFEval task](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/ifeval/ifeval.yaml)
- [GSM8K task](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/gsm8k/gsm8k.yaml)
