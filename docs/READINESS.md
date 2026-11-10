# Production readiness register

**Assessment date:** 2026-11-09. **Engine/source baseline:** `507661b0e1271e4f3757e824aa1fe1b7c3baf811`.

**Decision: not yet production-qualified as a whole.** There is substantial kernel, integration, real-model, and historical hardware evidence. The remaining launch work is chiefly release-specific qualification, sustained/failure-path validation, explicit support limits, and contributor/release discipline—not proof that every feature is broken or needs rewriting.

This is an evidence audit, not a security certification. Grades below are **provisional**. No A grade or production approval is awarded by the inventory, a green build, or a historical benchmark. See [release gates](#release-gates), the executable [release checklist](RELEASE_CHECKLIST.md), and [contributor requirements](../CONTRIBUTING.md).

## Scope and grading

The [feature inventory](FEATURES.md) retains its 22 sections. Stable IDs below partition every section into observable contracts; detailed implementation bullets belong to the corresponding row. Do not renumber IDs when adding capabilities. Split a row when its scopes acquire materially different evidence or deployment status.

| Grade | Meaning | Rule |
|---|---|---|
| A | Production-qualified | All applicable release gates pass for a pinned configuration, with retained evidence and operator sign-off. |
| B | Controlled-deployment candidate | Implementation, focused regression coverage, and integration/historical evidence support a bounded candidate; qualification gaps remain. Not permission to deploy without the release checklist. |
| C | Experimental / partially qualified | Meaningful implementation or test evidence exists, but breadth, quality, failure handling, or system qualification is insufficient for the claimed scope. |
| D | Known deployment blocker | A reproduced failure violates the declared contract in that scope. Attach reproduction and affected configurations. |
| U | Unassessed | Evidence is insufficient to judge the specific configuration or contract. Not a failing test. |

Grades are gates, not weighted averages. A correctness, isolation, corruption, or deadlock blocker cannot be offset by throughput. Deployment readiness is bounded by its critical dependencies and interactions. A candidate may omit an optional C/U capability only if its exclusion is enforced and documented; it cannot waive a required capability by calling it experimental. No D findings were established by this audit; that is not evidence of absence of defects.

### Six assessment dimensions

Use **Pass / Partial / Fail / Unknown / N/A** in release records. Pass requires a dated, scoped result; test existence alone means Partial at most. N/A requires a rationale. The register records evidence and promotion gaps rather than manufacturing numerical scores.

| Dimension | Audit-level evaluation | Evidence and remaining work |
|---|---|---|
| Correctness / quality | Partial | Extensive reference, rollback, kernel and real-model tests; distinguish exact tokens, tensor tolerances, quantization quality, and stochastic marginal correctness. G1/G2/G5. |
| Compatibility / coverage | Partial | Historical Ampere/Ada, PCIe/NVLink and TP=1/2/4 runs exist. They do not cover every current model × quant × driver × topology × context combination. G0/G1/G2/G6. |
| Reliability / recovery | Partial | Live disconnect, KV pressure, shutdown, and worker-error tests exist. Sustained churn, failure/restart exercises, and release-duration soak evidence remain open. G3/G4. |
| Security / isolation | Partial | Auth, scoped Responses, grammar/parser adversarial cases, image limits and SSRF policy tests exist. Deployment trust boundary, abuse budgets, and cross-request isolation need release review. G4. |
| Performance / capacity | Partial | Historical microbenchmarks and HTTP campaigns are substantial. No approved launch SLO/capacity envelope or current sustained-load receipt is attached. G0/G3/G5. |
| Operability / regression protection | Partial | Metrics, packaging, CPU CI, GPU test tasks, and historical receipts exist. Health semantics, recovery/rollback, GPU release evidence, and ownership need sign-off. G6/G7. |

These are project-level audit results, not six passing scores inherited by every row. Each selected release configuration must fill all six dimensions separately in its receipt. Unknown combinations stay U even where the capability's default candidate grade is B.

## Evidence ledger

Paths beginning `tests/` in the register mean `crates/w34-host/tests/`; source paths begin `crates/w34-host/src/` unless stated otherwise. Test files establish executable coverage, not that every case ran in the latest check.

| Ref | Evidence | Interpretation |
|---|---|---|
| E1 | [STATE.md](STATE.md), latest MoE consolidation and Flash-Next parity entries | Historical reports of full checks and explicit real-checkpoint TP1/TP2 parity. Recent entries supersede earlier resolved failures; do not resurrect the old B=5 handoff failure. |
| E2 | [BENCHMARK_AUDIT.md](BENCHMARK_AUDIT.md) | Pinned post-merge 26-configuration audit, HTTP measurements, and documented residual 9B performance regression. Local raw-evidence location is not automatically available to contributors. |
| E3 | [STATE.md](STATE.md), hardware qualification and NVLink P2P / TP=4 sections | Historical 3060 Ti / 4060 Ti / 4090 and 4×3090 NVLink results. Not fresh qualification of this baseline. |
| E4 | [HGPTQ.md](HGPTQ.md), [MODELS.md](MODELS.md), [QWEN_FLASH_NEXT.md](QWEN_FLASH_NEXT.md) | Architecture, checkpoint, and format scope; combine with execution evidence rather than treating support documentation as a pass. |
| E5 | [STRUCTURED_OUTPUTS.md](STRUCTURED_OUTPUTS.md), [GENERATIVE_APIS.md](GENERATIVE_APIS.md), [DECISION_MODELS.md](DECISION_MODELS.md), [SERVING.md](SERVING.md) | Public behavior contracts and limits; compatibility is with documented behavior, not the entirety of upstream APIs. |
| E6 | [Performance campaigns](../benchmarks/perf/README.md), [quality qualification](../benchmarks/quality/QUALIFICATION.md), [automation](../benchmarks/quality/AUTOMATION.md) | Retained historical measurements/receipts. Small `lm-eval --limit` runs qualify tooling, not broad model quality. |
| E7 | [.mise.toml](../.mise.toml), [test prerequisites](../tools/prep_test_models.py), [skip helpers](../crates/w34-host/tests/common/mod.rs), [CI](../.github/workflows/docker-image.yml) | `check` runs CUDA release tests; strict mode catches helper-based skips. Ignored tests, low-VRAM branches, and topology-specific branches still need explicit accounting. CPU oracle tests do not run live vLLM/SGLang comparisons. |
| E8 | [Packaging experiment](CUBLAS_PACKAGING_EXPERIMENT.md), [Dockerfile](../Dockerfile), E3 | Shared cuBLAS is the chosen default; older image sizes and Ada runs belong to their recorded binaries. |
| E9 | Fresh assessment verification | Core `prep_test_models.py --check` passed on 2026-11-09 (15 tracked fixtures and four core snapshot entries). Strict two-3090-Ti `mise run check` result is recorded in the release checklist. No fresh external-oracle, Ada, TP4, container, or soak qualification is implied. |

## Capability register

**Next gates** are required to promote the row for an in-scope deployment, not allegations of missing implementation. All rows also require G0 (scope) and G7 (receipt/sign-off). B rows require requalification; C rows need explicit scope-specific evidence before promotion. Historical tests are deliberately credited rather than discarded for not running during this audit.

### Surface and deployment — inventory §§1–4, 21–22

| ID | Inventory scope / contract | Grade | Evidence anchors | Promotion gap / next gates |
|---|---|---|---|---|
| CLI-01 | §1 dispatch, flags, validation, local/Hub/offline operation | B | `src/cli/{options,tests}.rs`; `tests/{cli_options,serving_options}.rs`; E4 | Exact launch recipe, pinned revisions, missing-cache/invalid-option behavior. G1/G6. |
| CLI-02 | §1 bench, dump/greedy/profile-diff, chat reporting | B | `src/cli/{commands,oracle}.rs`; `tools/oracle/`; E2/E7 | Verify result semantics and retain raw outputs; tool tests are not live oracle runs. G2/G5. |
| API-01 | §2 Chat/legacy completions, SSE, usage, reasoning/tool deltas | B | `tests/{axum_serving,generative_apis,http_constraints}.rs`; `src/server/http.rs`; E5 | Pin supported client versions; validate terminal/usage/error events and slow/disconnecting clients. G2/G3/G4. |
| API-02 | §2 Responses create/stream/get/delete/cancel/count, chaining and scoped store | B | `src/responses_api{,/tests}.rs`; `tests/generative_apis.rs`; E5 | Credential-scope negatives, eviction/concurrent cancel races, restart semantics. In-memory store is not durable history. G2/G4/G6. |
| API-03 | §2 Anthropic Messages/thinking/tools/count and billing-header normalization | B | `src/anthropic_api.rs`; `tests/{generative_apis,http_constraints}.rs`; E5 | Real supported-client conversation replay; malformed budgets/tool results and streaming event ordering. G2/G4. |
| API-04 | §2 preprocessing/admission, body limits/timeouts, backpressure | B | `src/server/http.rs` (`BODY_LIMIT`, `BODY_TIMEOUT`); `tests/axum_serving.rs` | Burst/slow-body/CPU-heavy preprocessing mixed with decode; bounded memory and recovery under overload. G3/G4. |
| API-05 | §2 model discovery, tokenization and token-count routes | B | `tests/{generative_apis,serving_options}.rs`; `src/server/http.rs` | Alias/count consistency with actual templates/images; auth and admission on aliases. G2/G4. |
| OPS-01 | §2/§21 health, Prometheus and JSON telemetry | B | `src/{telemetry,server/http}.rs`; [METRICS.md](METRICS.md) | `/health` checks worker panic and stopping, not arbitrary GPU forward progress. Demonstrate stalled-worker detection externally or implement/test it; reconcile counters under cancel/error. G3/G6. |
| DEC-01 | §3 Clef joint head, Kev pointer head | B | `tests/decision_models.rs`; `src/{clef,decision}/`; E5/E7 | Synthetic Kev-head coverage is not qualification of every production head. Pin real checkpoints/head artifacts and golden results. G2/G3. |
| DEC-02 | §3 Nimble, Tev1, OpenJev, Xor/Lev scoring | C | `src/decision/{scoring,tests}.rs`; `tests/decision_models.rs`; E5 | Per-profile real-model correctness and throughput; option-count/normalization boundaries. Scores are not automatically calibrated probabilities. G2/G5. |
| SEC-01 | §4 digest-based auth, Bearer/x-api-key route protection | B | `src/api_auth.rs`; `src/server/http.rs` auth tests; E5 | Auth is opt-in and service-level, not a tenant identity system. Verify fail-closed configuration and deployment exposure. G4/G6. |
| UI-01 | §4 embedded dashboard, CSP, cache headers and asset paths | B | `src/dashboard.rs::embedded_dashboard_and_assets_serve_without_filesystem`; `dashboard/`; E8 | Browser smoke against shipped image; metrics/dashboard are public at application level. Restrict at ingress if sensitive. G4/G6. |
| VER-01 | §21 tensor/token/profile oracle framework | B | `tools/oracle/`; `tests/xgrammar_oracle.rs`; E1/E2/E6/E7 | Fresh matched-checkpoint oracle receipts; distinguish exact matches from adjudicated numerical divergence. G2/G5. |
| PKG-01 | §22 Docker runtime, non-root user, shared CUDA libs, caches/healthcheck | B | `Dockerfile`; E3/E8 | Pull/run exact image digest, read-only/offline model-cache smoke, driver floor, shutdown/restart and upgrade rollback. G6. |
| PKG-02 | §22 sm_86/sm_89 native fatbinary and runtime device properties | B | `crates/w34-kernels/build.rs`; `src/cuda.rs`; E3/E8 | Native-code smoke on both claimed architectures at release revision; PTX fallback separately if advertised. G1/G6. |

### Request semantics — inventory §§5–6, 11, 20

| ID | Inventory scope / contract | Grade | Evidence anchors | Promotion gap / next gates |
|---|---|---|---|---|
| GRM-01 | §5 schema/JSON/regex/Lark/EBNF/choice compilation and cache | B | `tests/{structured_constraints,xgrammar_oracle}.rs`; `src/constraints.rs`; E5 | Supported-schema boundaries, costly/recursive grammar budgets, concurrent cache churn. Do not infer per-tenant quotas from the shared cache limit. G2/G4. |
| GRM-02 | §5 GPU masks, reasoning gate, stop/EOS handling, speculative rollback | B | `tests/{structured_constraints,http_constraints,sampling_policy}.rs`; E1 | Grammar × forced rejection × EOS × cancellation interaction on every enabled speculative mode. G2/G4. |
| PAR-01 | §6 canonical Qwen XML tool choice and typed parameter extraction | B | `tests/structured_constraints.rs` injection/envelope regressions; `src/tool_constraints.rs`; E5 | Canonical-envelope interoperability and adversarial streaming replay. Absence of a schema `strict` flag does not disable constraint enforcement. G2/G4. |
| PAR-02 | §6 incremental content/reasoning parsing, implicit/duplicate tag handling | B | `src/parser.rs` tests; `tests/{generative_apis,http_constraints}.rs`; E1 | Delimiter splits, incomplete envelopes and endpoint event consistency. Preserve documented parsing semantics unless a reproduced issue warrants change. G2/G4. |
| VIS-01 | §11 image decode/resize/normalize, ViT, positions, merger and vision FA2 | B | `tests/{image_decode_limits,jpeg_decode,vision_positions,vision_rounding,tq4_vision_and_moe}.rs`; E1/E3 | Real-model oracle at supported image sizes/counts; bounded expanded tokens and peak memory. G2/G3. |
| VIS-02 | §11 content-keyed multimodal prefix reuse | B | `src/scheduled_vision.rs`; vision benchmark isolation checks; E1/E3 | Image/text prefix collision threat model and repeated eviction/reuse checks. Noncryptographic FNV hashing is not proof of adversarial collision resistance. G2/G4. |
| VIS-03 | §11 opt-in remote image retrieval and SSRF limits | B | `src/remote_images.rs` transport/IP/deadline tests; `tests/multimodal_images.rs` | Validate DNS-to-connection binding, address policy, proxy/redirect rules in deployment; enforce egress restrictions. Remote fetch stays off unless qualified. G4. |
| SMP-01 | §20 grammar/bias/penalties/filtering, greedy and logprobs | B | `tests/{gpu_sampler,sampling_policy}.rs`; `op_sampler.cu`; E1 | Full sampler/fast-path parity with masks, nonfinite/empty distributions, graph padding and supported vocabulary sizes. G2. |
| SMP-02 | §20 stochastic selection and speculative residual distribution | C | `tests/{rejection_sampler,gpu_rejection_sampler,gpu_sampler}.rs`; `src/parser.rs::uniform_f32` | Host/GPU seam and enumerated marginal tests exist. Qualify end-to-end distribution across configured policies, seeds, rejection positions and batching; do not demand identical sampled strings as proof. G2. |

### Engine state and concurrency — inventory §§7–10, 12, 19

| ID | Inventory scope / contract | Grade | Evidence anchors | Promotion gap / next gates |
|---|---|---|---|---|
| ENG-01 | §7 scheduling, aligned chunked prefill, mixed decode, scratch sizing | B | `src/scheduler.rs`; `tests/{radix_cache_and_scheduler,axum_serving}.rs`; E1/E2 | Mixed-length fairness, queue/latency limits and capacity at advertised context/concurrency. G3. |
| ENG-02 | §7 bounded ingress, cancellation, pipelined decode, rank transaction | B | `src/{engine_worker,coordinated}.rs`; `tests/{axum_serving,decode_graphs}.rs` | Cancel during queued/prefill/decode/verify stages; rank failure, output backpressure, drain and resource reuse. G3/G4. |
| ENG-03 | §8 graph warmup/capture/replay, valid-row isolation | B | `tests/{decode_graphs,collectives_and_gdn_wy,speculative_decoding}.rs`; E1/E3 | Fresh-process startup repetitions, all selected graph buckets, eager parity, padded state and sampler isolation. G1/G2/G3. |
| MEM-01 | §8 arenas, host embedding offload and capacity planning | B | `src/model/{rank,tests}.rs`; E3 8-GB qualification | VRAM/RAM/pinned-memory high-water bounds, insufficient-memory errors, same release on selected low-memory SKUs. G3/G6. |
| CACHE-01 | §9 radix dual-lock/dual-LRU/tombstone and split semantics | B | `tests/radix_cache_and_scheduler.rs::test_mamba_radix_cache_dual_lock_dual_lru_tombstone_and_split`; E1 | Long-running split/evict/revive churn and ownership conservation, not just cache-hit speed. G3/G4. |
| CACHE-02 | §9 ping-pong checkpoint donation and warm target/draft reuse | B | `tests/{radix_cache_and_scheduler,speculative_decoding,qwen4_flash_next}.rs`; E1 | Reject across checkpoint boundaries, cancel before donation, image isolation and warm/cold parity under pressure. G2/G3/G4. |
| MEM-02 | §10 typed KV layouts, page allocation and rank clamping | B | `src/cache.rs`; `tests/backbone_and_collectives.rs::test_fragmented_paged_attention_and_tq4_strides` | Exhaustion/eviction at supported limits; no aliasing or stale cross-request reads. G2/G3/G4. |
| MEM-03 | §10 GDN state pools, StateTx phases and speculative slot reclamation | B | `src/cache.rs` tests; `tests/rejection_sampler.rs`; `tests/backbone_and_collectives.rs` | Repeated reject/cancel/exhaust/reuse cycles and failure atomicity; reconcile available slots after quiescence. G2/G3/G4. |
| SPEC-01 | §12 MTP predictor, optional online proposal-head quantization, four slots | B | `tests/{mtp_proposer,speculative_handoff,quant_formats}.rs`; E1/E2 | K=1..3, B=1..4 and B=5 fallback; proposal-head memory decisions; pinned model-specific parity. G2/G3. |
| SPEC-02 | §12 BF16 DFlash2, block verification, private draft context and selector | B | `tests/{dflash_proposer,dflash_draft_kernels,dflash_bf16_attention}.rs`; E1/E3 | Model-specific M=8/16, warm prefixes, drafter fit and no-state-slot failures; scope B to demonstrated greedy paths. G2/G3. |
| SPEC-03 | §12 four-length commit/rollback and dynamic handoff | B | `tests/{rejection_sampler,speculative_decoding,speculative_handoff}.rs`; E1 | Sustained late-peer churn, stop/EOS, constraints and cancellation; stochastic scope also depends on SMP-02. G2/G3/G4. |
| TP-01 | §19 TP2 mapped-host collectives, typed payloads and DMA crossover | B | `tests/collectives_and_gdn_wy.rs`; `src/cuda/tests.rs`; E1/E3 | Actual topology, small/large payload boundaries, cold capture and failure behavior. G1/G2/G3. |
| TP-02 | §19 TP2 P2P/NVLink and fallback selection | B | `tests/collectives_and_gdn_wy.rs::test_tp2_p2p_and_tp4_collectives`; E3 | Historical evidence exists; current local PCIe run cannot qualify NVLink. Assert selected transport on real hardware. G1/G2/G6. |
| TP-03 | §19 TP4/EP4 collectives and execution | C | `tests/collectives_and_gdn_wy.rs`; E3 4×3090 run | Historical implementation and execution, not U or broken. Requalify current four-rank model/graph/speculation paths; count branch skips explicitly. G1/G2/G3. |

### Model and kernel foundation — inventory §§13–18

| ID | Inventory scope / contract | Grade | Evidence anchors | Promotion gap / next gates |
|---|---|---|---|---|
| MOD-01 | §13 Qwen dense model spine and precision policy | B | `tests/backbone_and_collectives.rs`; E1/E2/E3/E4 | B applies to evidenced checkpoints, not every size/revision (e.g. no automatic 0.8B promotion). Pin model/quant/context and oracle tolerances. G1/G2/G5. |
| MOD-02 | §13 35B-A3B routed/shared-expert model | B | `tests/moe_sparse.rs`; E1/E2/E4 | Real-weight TP/mode and long-context quality at release revision; distinguish numerical adjudication from exact token parity. G2/G5. |
| MOD-03 | §13 Flash-Next HC/PLE/QSA and precision policy | B | `tests/qwen4_flash_next.rs`; E1/E3/E4 | Explicit ignored real-checkpoint tests required; synthetic coverage alone is insufficient. Recent short exact-parity runs do not qualify all contexts. G1/G2/G5. |
| LOAD-01 | §14 mmap/shards/Hub, FP16/BF16/GPTQ/HGPTQ ingestion | B | `src/loader/`; `tests/{hgptq_checkpoint,backbone_and_collectives}.rs`; E4/E7 | Offline/revision/error paths, malformed/truncated checkpoints, host-memory startup budget. G1/G4/G6. |
| LOAD-02 | §14 AWQ/AutoRound/compressed-tensors variant breadth and act-order | C | `tests/quant_formats.rs::cached_hf_quantized_and_unquantized_checkpoints`; E4 | Explicitly ignored end-to-end format matrix must run for advertised variants. Format parser support is not checkpoint qualification. G1/G2. |
| GEMM-01 | §14 Marlin repack/GEMMs, grouping/zero-points and reduction invariance | B | `tests/quant_formats.rs` scalar/identical-row tests; E1/E2 | Qualify actual format/shape dispatch; do not infer W4A8 or every act-order combination from W4A16/W8 reference tests. G2/G5. |
| MOE-01 | §15 expert packing, hot-cache LRU, mapped pulls and streamed prefill | B | `src/moe_offload/tests.rs`; `tests/{moe_sparse,qwen4_flash_next}.rs`; E1 | Cache churn at regime boundaries, fully resident vs tiered paths, small VRAM and long-prefill coexistence. G2/G3. |
| MOE-02 | §15 CPU bridge, SIMD dispatch, mailbox and async accumulation | B | `src/moe_cpu_bridge/`; E1/E3; `tools/check_cpu_simd.py` | AVX2/AVX512/portable numeric coverage plus worker failure and pinned-memory lifetime stress. Assembly inspection alone is not numerical validation. G2/G3/G6. |
| GDN-01 | §16 conv/recurrent decode/verify and FP32 state | B | `tests/{gdn_conv_rounding,backbone_and_collectives}.rs`; E1 | Explicit input/output slots, continuation and reject sweeps on real selected model. G2. |
| GDN-02 | §16 WY/chunk64 prefill, fallback and continuation | B | `tests/{collectives_and_gdn_wy,backbone_and_collectives}.rs`; E1 | Tile boundaries, mixed batches and documented tolerances; avoid universal 0-ULP claims. G2/G3. |
| ATT-01 | §17 TQ4/TQ4-all/Q8 layouts, rotations/codebook and compression | B | `tests/{tq4_vision_and_moe,backbone_and_collectives}.rs`; E1/E4 | Layout/compression correctness is distinct from model quality. Pair every deployed quant policy with ATT-03. G2/G5. |
| ATT-02 | §17 paged decode, MMA/Split-KV and prefill FA2 | B | `tests/paged_fa2.rs`; `tests/backbone_and_collectives.rs`; E1/E3 | Fragmented/long contexts, variable batches, graph/eager parity and head dimensions. G2/G3. |
| ATT-03 | §17 quantized-KV quality and advertised long-context capacity | C | E1 historical 32k–262k needle and 252k HTTP runs; E6 | Meaningful historical evidence, but not universal losslessness. Representative task quality, actual memory envelope and latency SLO at maximum advertised context. G3/G5. |
| OPS-02 | §18 fused norms/RoPE/gates/activations and small-N GEMMs | B | `tests/{gemma_residual_rounding,target_rope_cache_rounding,moe_sparse,quant_formats}.rs`; E1/E2 | Shape/rounding dispatch regressions and model-level plain/spec parity; performance baseline for changed kernels. G2/G5. |

## Release gates

**All gates are OPEN until a selected release receipt closes them.** A successful audit/check closes only its recorded substeps. Assign a named owner in the checklist; role labels below are suggested ownership, not claims that someone accepted the work.

| Gate | Priority / suggested owner | Acceptance criteria |
|---|---|---|
| G0 Scope and SLO freeze | P0 / release maintainer | Pin binary/image, models/drafters, GPU/topology, driver/toolchain, formats, APIs, context/concurrency, trust boundary and absolute latency/error/memory budgets. Explicitly exclude everything not qualified. |
| G1 Reproducible baseline | P0 / runtime maintainer | Prerequisites verified; strict full `check` green; list every ignored/skipped branch; run relevant ignored real-model/format tests; reproduce on every claimed hardware family/topology. Missing hardware means restrict scope, not infer a pass. |
| G2 Correctness and integration | P0 / subsystem maintainers | Documented tensor tolerances and exact token contracts pass; cold/warm, eager/graph, plain/spec and cross-request state tests pass. Real clients exercise advertised wire contracts. Stochastic modes additionally demonstrate target marginal correctness with predeclared statistical tolerance and enough independent draws. |
| G3 Load, pressure and soak | P0 / runtime + operations | Proposed minimum: 8 hours and 10,000 completed requests (both), mixed prompts/images/tools as enabled; sustained selected concurrency plus a bounded 2× overload burst. Zero crashes/deadlocks/unexplained errors, no unbounded RAM/VRAM/pinned-memory growth after cache warmup, ownership conserved after cancel/drain, and G0 SLOs met. Expected overload rejections counted separately. Maintainer must approve or strengthen this minimum before execution. |
| G4 Security and failure boundaries | P0 / serving maintainer | Auth/aliases and Responses scope negative tests; image/grammar/body abuse limits; cross-request/cache isolation; cancellation at every phase. Inject worker/CPU-bridge failures safely in tests or disposable processes; demonstrate requests terminate and supervisor restarts within G0 budget. Define public metrics exposure and ingress/egress controls. |
| G5 Quality and performance | P0 for advertised claims / model maintainer | Compare selected KV/spec/model modes on representative tasks and long-context retrieval, not only smoke prompts. Freeze acceptance tolerance before measuring. Run matched performance baselines; adjudicate known 9B regression rather than erase it. Competitor profiling is required for competitor superiority claims, not for basic service safety. |
| G6 Deployment and recovery | P0 / operations | Exact image digest starts under intended user/driver/cache permissions, serves a real request, drains/restarts, and rolls back to last known-good digest. Probe/alert stalled progress, test metrics, document non-durable Responses history and key rotation. Keep artifact and checkpoint hashes. |
| G7 Contribution and release record | P0 / release maintainer | Named owners, required CI plus GPU receipts for GPU changes, reproduction template, no weakened assertions, immutable release evidence and signed go/no-go. Open C/U scopes are disabled or clearly outside the supported configuration; no in-scope D blocker. |

### High-risk interaction qualification

| Interaction | Required invariant | Existing starting point |
|---|---|---|
| Speculation × constraints × stop/EOS × rejection | No rejected tokens persist in target/grammar state or client output; valid terminal events | `tests/{http_constraints,rejection_sampler,speculative_decoding}.rs` |
| Cache × vision × eviction × cancellation | Correct image identity, cold/warm output contract, balanced pages/state slots after quiescence | `tests/{tq4_vision_and_moe,radix_cache_and_scheduler,axum_serving}.rs` |
| Batching × long prefill × late-peer handoff | No starvation, token loss or invalid single-stream-vs-batched oracle assumptions | `tests/{speculative_handoff,decode_graphs,axum_serving}.rs` |
| TP × cold graph capture × rank/host failure | No lazy-load hangs; consistent transaction or bounded process failure/restart | `tests/collectives_and_gdn_wy.rs`; `src/coordinated.rs`; worker supervision tests |
| MoE offload × memory pressure × speculative verify | No stale mailbox reads, in-flight slot eviction, pinned-buffer reuse or leaked ownership | `src/moe_offload/tests.rs`; `tests/qwen4_flash_next.rs`; E1 |
| Streaming tools × partial delimiters × slow/disconnected clients | No incomplete executable tool call, bounded buffers, prompt resource reclamation | `tests/{structured_constraints,http_constraints,axum_serving}.rs` |
| Responses chaining × credentials × cancel/eviction | No cross-scope retrieval or contradictory terminal lifecycle; defined expiry/restart behavior | `src/responses_api/tests.rs`; `tests/generative_apis.rs` |

## Focused lifecycle hardening follow-up

The `fix/production-lifecycle-hardening` round (based on `bc2a3e5`) addresses specific findings without replacing the scheduler, speculative transaction model, or protocol adapters:

| IDs | Change / regression coverage | Remaining qualification |
|---|---|---|
| ENG-02 / MEM-03 | Deferred pipeline release records are drained after abort on a failed next-step wait; CPU regressions check abort-before-release, exactly-once delivery, and original-error preservation. | Real device/rank failure and long-running pool conservation under churn. |
| API-02 | One Responses-store lock arbitrates explicit cancellation and terminal publication; streaming/sync failure paths preserve that winner. Scope, racing publication, transport-vs-explicit cancellation and live SSE/store agreement are tested. | Sustained lifecycle/eviction churn and external client compatibility with the documented `response.cancelled` extension. |
| GRM-02 / SPEC-03 | Native sampled generation stops at terminal grammar state even when the terminal ID is not a sampling stop/EOS ID; CPU predicate and real-model native MTP regressions. | Sampled stateful-grammar late-peer handoff remains unqualified. Deterministic request-specific handoff synchronization is not exposed; no timing-dependent test or broad hook was added in this round. |
| OPS-01 | `/health` reports stopping as unavailable while retaining panic diagnostics; CPU and live route checks cover shutdown readiness. | A responsive probe still does not prove GPU progress. External hard-stop/restart rehearsal remains an operator gate; see [SERVING.md](SERVING.md#readiness-shutdown-and-bounded-recovery). |
| MEM-01 | Legacy pool budgeting charges allocated arena capacity, not only its used bytes; state/KV boundary regressions cover the difference. Live serving-budget arithmetic and intentional minimum MoE-plan fallback remain unchanged. | Actual peak-memory/near-capacity qualification across supported configurations. |

These changes do not promote any row to A or close G0–G7 automatically. See [STATE.md](STATE.md) for the completed verification results and limitations.

## Extended-workload execution support

[`w34perf soak`](../benchmarks/perf/SOAK.md) now supplies a bounded driver for duration/completion targets, sustained/burst replay, local disconnect churn, completion-progress checks, optional endpoint RSS/GPU growth gates and a final inference probe. The checked-in coding workload is synthetic/replay; a short real-model smoke verifies the driver, **not G3 qualification**. Missing image/grammar/Responses-cancel coverage, server ownership conservation, production workload representativeness and external failure/recovery qualification remain explicit. See [STATE.md](STATE.md) for the smoke receipt and limitations; no readiness grades or gates are promoted by adding the harness.

## Assessment maintenance

For every promotion, attach: ID, scope, six dimension results, exact source/image/checkpoint revisions, command and exit status, actual executed/skipped tests, hardware/software metadata, artifact location/hash, observed limits, remaining risks, owner and date. Keep raw failed runs as well as successful ones. Large evidence belongs in durable release/CI storage with a checked-in manifest; `/tmp` or an agent-local `outputs/` path is not a contributor-accessible release archive.

Relevant code, checkpoint, compiler, driver, topology or workload changes mark affected qualification **stale** until reassessed. Preserve the prior receipt rather than relabeling it. A regression report must identify the readiness ID and affected configuration; a fix needs a failing regression first, restored tests, and an updated receipt. Start from [CONTRIBUTING.md](../CONTRIBUTING.md).
