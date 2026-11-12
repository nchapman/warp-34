# Model acquisition

`w34` accepts a local checkpoint directory or a Hugging Face repository ID:

```sh
# Download on first use, then load the model. No Python/CLI dependency.
w34 serve --model nchapman/Qwen3.5-4B-W4A16-HGPTQ

# Reproducible deployment: use a full 40-character commit SHA.
w34 serve --model username/modelname --revision <commit-sha>

# No network access or cache writes; also works with read-only cache mounts.
w34 serve --model nchapman/Qwen3.5-4B-W4A16-HGPTQ --offline

# Local directories never contact the Hub.
w34 generate --model /models/qwen --prompt 'Hello'
```

Acquisition is shared by `generate`, `bench`, `greedy-diff`, `dump-diff`, `serve`,
and `chat`. Live `profile-diff` forwards these options to its nested `bench`;
comparison of saved profiles does not resolve or download models and ignores
Hub acquisition options. Argument
validation and acquisition happen before CUDA initialization. The original Hub ID
remains the public API model name and the model identifier in benchmark reports.

## Revisions, cache, and offline mode

- `--revision` accepts a branch, tag, or full commit SHA; the default is `main`.
  Every online invocation resolves the requested ref and checks the repository
  listing. Cached file payloads are reused; **online startup still needs network
  access and a writable cache**, even when weights are already present.
- Metadata and weights are fetched by the **same immutable commit**, not by a
  branch that could change midway through downloading. The resolved commit is
  printed to stderr. A mutable cache ref is published atomically only after the
  selected checkpoint is complete and validated.
- `--offline` or `HF_HUB_OFFLINE=1` forbids all Hub access. A named revision must
  have an exact cached `refs/<revision>` mapping; a commit SHA selects its exact
  snapshot. There is no fallback to an arbitrary directory under `snapshots/`.
- Local directories remain local. Passing a revision alongside a local directory
  is an error rather than silently ignoring the revision.
- The native client is pinned to Hugging Face's `hf-hub` 1.0.0, including its native
  Xet transfer support. It uses the standard `models--owner--repo/{blobs,refs,snapshots}`
  layout, interoperable with `hf download` / Python `huggingface_hub`.
- warp-34 additionally records the required file list under each repository's
  `.w34/<commit>.json`. This is bookkeeping, not another copy of model weights.
  Offline startup checks every recorded file. Existing caches made by other tools
  without that manifest are accepted after checking configuration, tokenizer,
  available decision heads, Safetensors headers, and every indexed shard. Without
  network metadata, optional upstream files absent from such a cache cannot be
  discovered; use one online acquisition to record the full selected file set.
- HTTP transport, authentication, redirects, retries, locking, and Xet transfers
  use the released SDK's behavior without a custom HTTP client or endpoint policy.
  The SDK does not configure HTTP request timeouts, so a stalled endpoint can
  block online startup. Use offline mode when startup must not depend on Hub
  availability.
- An interrupted acquisition fails rather than returning a partial checkpoint.
  Retry the same command to reuse completed cached files. Disk and permission
  errors are surfaced; warp-34 does not delete or garbage-collect existing cache
  entries. No promise of resuming every individual file at the last byte is made.

### DFlash2

Target and draft repositories have **independent revisions**:

```sh
w34 serve --model nchapman/Qwen3.5-4B-W4A16-HGPTQ \
  --revision <target-commit> \
  --speculative dflash2 --draft-model nchapman/Qwen3.5-4B-DFlash2 \
  --draft-revision <draft-commit>
```

Both default to their own `main`; the target's revision is never applied to the
unrelated draft repository. `--offline` applies to both. Drafters may omit a
separate tokenizer because the target supplies it. Native MTP shares target
weights and has no separate download or revision.

## Standard environment and authentication

| Variable | Behavior |
|---|---|
| `HF_HUB_CACHE` | Explicit Hub cache directory; overrides `HUGGINGFACE_HUB_CACHE` and `HF_HOME/hub` |
| `HUGGINGFACE_HUB_CACHE` | SDK-supported legacy cache-directory fallback |
| `HF_HOME` | HF state directory; defaults to `$XDG_CACHE_HOME/huggingface`, or `$HOME/.cache/huggingface` |
| `HF_TOKEN` | Authentication token; takes precedence over the saved login |
| `HF_TOKEN_PATH` | SDK checks this token file before falling back to `$HF_HOME/token` |
| `HF_HUB_OFFLINE` | Disable Hub access (`1`, `ON`, `YES`, or `TRUE`, case-insensitive) |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | SDK credential opt-out: any non-empty value disables lookup (including `0`); unset to enable |
| `HF_HUB_DISABLE_PROGRESS_BARS` | Suppress transfer progress; normal resolution/result messages remain |
| `HF_ENDPOINT` | Explicit Hub-compatible endpoint override; defaults to `https://huggingface.co` |

Authentication, endpoint, and cache environment settings are resolved by
`hf-hub` itself; warp-34 does not implement a second credential lookup or override
the SDK's transport policy. warp-34 maps `--offline`/`HF_HUB_OFFLINE` to the SDK's
`local_files_only` lookup and controls its own progress reporter.

An existing `hf auth login` is recognized, but installing that CLI is **optional**:
exporting `HF_TOKEN` is sufficient. Tokens are not accepted as CLI flags, written
to manifests, or included in download errors. Endpoint overrides are operator
configuration—only point `HF_ENDPOINT` at a service you trust with your credentials.

For gated models, accept the repository's license/access conditions on Hugging
Face and authenticate as an authorized user. Private/not-found, denied access,
missing revisions, network errors, cache locks, and filesystem failures have
separate diagnostic guidance. Offline Hub loads make no network requests;
local-path loads bypass the Hub client entirely.

## Downloaded assets and safety

The downloader selects the formats the existing runtime can actually load:

- Root model configuration, generation/quantization configuration, tokenizer
  assets, chat templates, and image/video processor metadata when present.
- `model.safetensors.index.json` and **exactly its referenced shards**, or
  `model.safetensors` for an unsharded checkpoint.
- `head.safetensors`, `joint_head.safetensors`, and `joint_head_config.json` when
  present, so decision-model heads are not silently omitted.

It does not fetch alternate checkpoints just because their names end in
`.safetensors`, nor pickle/PyTorch `.bin`, GGUF, training artifacts, or repository
Python code. Traversal/glob paths in shard indexes are rejected. Required files
cannot be silently skipped if the server returns a per-file 404.

Repository IDs do **not** imply arbitrary architecture support. Supported targets
are the existing Qwen 3.5/3.6/3.8 hybrid dense/MoE families; DFlash2 uses its explicit
drafter configuration path and native BF16/FP16 weights. Configuration validation
runs before downloading weights, and existing runtime geometry/quantization
checks still apply. There is no `trust_remote_code` switch and no repository code
execution.

## Supported architectures and quantization formats

| Category | Supported formats & configurations |
|---|---|
| **Architectures** | `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration` (dense `0.8B`, `4B`, `9B`, `27B`), `Qwen3_5MoeForCausalLM` / `Qwen3_5MoeForConditionalGeneration` (`35B-A3B` 256-expert MoE), `Qwen4ExpForCausalLM` (`Qwen3.8-Flash-Next` 512-expert MoE with `HyperConnection`, `QSA`, `PLE`, and `Qwen4MTP`), `DFlashDraftModel` (`4B`, `27B`), and System One decision heads (`Clef`, `Kev`, `Nimble`, `Tev1`, `OpenJev`, `Xor` / `Lev`). |
| **Weight Quantization** | `GPTQ` (`sym=true` and `sym=false`, `4-bit` and `8-bit`, `desc_act=false`), `AWQ` (`4-bit`, symmetric and asymmetric), `compressed-tensors` (`pack-quantized` `4-bit` and `8-bit`, symmetric and asymmetric), `AutoRound` (`4-bit`, symmetric and asymmetric), FP8 `PLE` (`F8_E4M3` + `BF16` per-row scale), and unquantized `FP16` / `BF16`. |
| **KV Cache Policies** | `--kv-format fp16` (16-bit baseline), `--kv-format q8_0` (`1.88×` compression, `272 B/row`), `--kv-format tq4` (`3.07×` hybrid: rotated `Q8_0` on the first 2 and last 2 full-attention layers + rotated LSQ `TQ4` `132 B/row` on interior layers), and `--kv-format tq4_all` (`3.88×` uniform `TQ4`). |
| **Multi-GPU & Offload** | `--tp 1`, `--tp 2` (NVLink P2P VRAM or pinned-sysmem PCIe), and `--tp 4` (`EP=4` 100% VRAM-resident 512-expert MoE); `--moe-offload auto|gpu-lru|cpu-hybrid` with three-tier 0-ULP CPU MoE evaluation (`AVX-512`, `AVX2+FMA+F16C`, or portable scalar). |

## Library callers and tests

`w34_host::hub::resolve_model` is the explicit acquisition API. The older
`loader::resolve_hf_model_dir` remains **cache-only** for library users and test
fixtures; missing large GPU fixtures never automatically download during tests.
Its cache root also respects the standard `HF_HOME`/`HF_HUB_CACHE` precedence.

Deterministic tests use the real Rust client against an in-process HTTP Hub
fixture, with isolated caches and explicit SDK options. Child-process tests
exercise the production entrypoint's SDK environment and saved-login behavior
without mutating the parent test process's environment. Tests cover revision pinning, filtering, reuse, credentials,
offline misses, partial snapshots, sharded weights, unsupported architectures,
and redacted access errors. Live Hub qualification is separate from the normal
suite and is not required for offline test runs.
