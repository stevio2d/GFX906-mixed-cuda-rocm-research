# Prompt For External AI: MiniMax-M2.7 llama.cpp Mixed CUDA+ROCm PP Optimization

Use this as a standalone technical brief. The goal is to find practical llama.cpp / ggml changes that can raise MiniMax-M2.7 prompt-processing throughput well beyond the current custom-fork result.

## Goal

Increase MiniMax-M2.7 prompt-processing throughput toward 1000 tokens/s while keeping:

- Model: MiniMax-M2.7 GGUF `UD-Q4_K_S`
- Context: `69376`
- KV cache: `q4_0` preferred for current production path, with Q8 KV also important to understand
- Host: mixed CUDA + ROCm/gfx906
- Runtime base: custom `/home/stefan/llama.cpp-gfx906` fork

Do not assume true AMD/NVIDIA GPU P2P exists. Treat the current fast cross-vendor path as host-staged, pinned-memory, overlapped copy.

## Hardware And Runtime

GPUs:

- 2 x RTX 3090, about 24 GiB each
- 2 x RTX 5070 Ti, about 16 GiB each
- 2 x gfx906 AMD MI50/MI60-class cards, about 32 GiB each

CPU/RAM:

- AMD EPYC 7532, 32 physical cores / 64 threads
- 8 populated DDR4-2666 RDIMM channels, 256 GiB total
- Best measured all-socket STREAM-style Triad about 105 GB/s
- CPU-only inference is far slower than GPU VRAM paths and is not a viable way to reach high PP for this target

Required ROCm/gfx906 environment:

```bash
export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCBLAS_LAYER=0
export ROCBLAS_LOG_LEVEL=0
export HIPBLASLT_LOG_LEVEL=0
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=8
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Do not set `HSA_ENABLE_SDMA=0` for mixed movement tests. It made mixed copy/tensor movement slower.

## Benchmark Invariants

Cold one-shot benchmarks remain diagnostic only. The primary target is now cache-aware OpenCode-style coding-session throughput:

- Use one long-lived `llama-server` process, preferably one slot.
- Enable prompt cache.
- Simulate repeated coding turns with a large shared prefix and smaller changed suffixes.
- Grow active context toward `69376`.
- Verify reuse from `timings.cache_n`, `tokens_cached`, prompt-cache logs, `/slots`, or equivalent evidence.
- Stop/reject runs when newly processed prompt throughput drops below `300 PP`; OpenCode should not be optimized around slower paths.
- Do not mix cold one-shot PP with cache-aware session PP in conclusions.

The main cache-aware benchmark script is:

```text
/home/stefan/llama.cpp-gfx906/bench-minimax-cache-session.sh
```

Current cache-aware baseline evidence:

```text
/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-20260606T003539Z
turn 1: context 34450, cache_n 0, processed prompt_n 34354, 334.4944 PP / 17.7698 TG
turn 2: context 45811, cache_n 34449, processed prompt_n 11266, 200.0801 processed PP / 14.7672 TG
turn 2 effective context throughput: 811.8821 context tokens/s
reuse evidence: timings.cache_n = 34449, tokens_cached = 45810
decision: stopped/rejected because newly processed PP is below the 300 PP OpenCode cutoff

/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-ls13137111117-ub800-20260607T035800Z
turn 1: context 34450, cache_n 0, processed prompt_n 34354, 334.3276 PP / 17.8378 TG
turn 2: context 45811, cache_n 34449, processed prompt_n 11266, 200.5461 processed PP / 14.4583 TG
decision: exact manual 13,13,7,11,11,7 split rerun; same failed ~200 PP cached-turn class

/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-graph-ub800-20260607T040100Z
turn 1: context 34450, cache_n 0, processed prompt_n 34354, 333.5311 PP / 17.7884 TG
turn 2: context 45811, cache_n 34449, processed prompt_n 11266, 199.3829 processed PP / 14.5173 TG
decision: exact auto-graph rerun of the cold-best ub800 q4-direct path; still fails the 300 PP cached-turn cutoff
```

Use the same cold diagnostic prompt payload when comparing legacy PP/TG:

```text
/home/stefan/llama.cpp-gfx906/bench-results/minimax-fattn-tmp-cap-20260604T040506Z/payload.json
```

Common production command shape:

```bash
/home/stefan/llama.cpp-gfx906/build-cuda-hip-dl-gfx906/bin/llama-server \
  -m /home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf \
  -c 69376 -ngl 999 -sm layer \
  -dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2 \
  -fa on -np 1 -fit off --numa distribute --no-warmup --jinja \
  -t 24 -tb 48 -lv 3 \
  -ctk q4_0 -ctv q4_0
```

Validation requirements:

- Record prompt eval throughput (PP), token generation throughput (TG), prompt tokens, output tokens, and output hash/content sanity.
- Reject runs with CPU fallback, no pipeline parallelism when pipeline is expected, OOM, request abort, or deterministic prompt-echo failures.
- Keep model, context, KV type, prompt, output length, and GPU residency identical when comparing.

## Current Best Result

Best confirmed custom-fork result:

```text
558.7603 PP / 31.2581 TG
repeat: 558.6422 PP / 31.2530 TG
prompt tokens: 11521
output tokens: 128
accepted hash prefix: 9bd2eb...
```

Best command additions over the common command:

```bash
-b 800 -ub 800 \
--placement-policy memory \
--device-speed 0.55,0.55,1.0,1.0,1.0,1.0 \
--cuda-fattn-no-stream-k \
--cuda-fattn-vec-qkv-headroom-mib 0 \
--reserve-pp-outputs 1 \
--layer-split 14,12,7,11,11,7 \
--output-device CUDA3 \
--cuda-fattn-mma-q4-0-direct \
--cuda-fattn-mma-q4-0-direct-mode all
```

Environment for the exact benchmark prompt with `ub800`:

```bash
export GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS=hip
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=321
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=321
export GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=1
```

Prompt-length-portable validation used `Q_MIN=2,Q_MAX=799` instead of exact `Q=321`.

Best artifacts:

```text
/home/stefan/llama.cpp-gfx906/bench-results/minimax-ub800-moe-reuse-enabled-fixed-20260605T081107Z
/home/stefan/llama.cpp-gfx906/bench-results/minimax-ub800-moe-reuse-enabled-fixed-repeat-20260605T081217Z
```

## Latest Upstream Baseline

Fresh upstream checkout:

```text
/home/stefan/llama.cpp-upstream-minimax-clean
commit 308f61c opencl: improve get_rows, cpy, concat and q6_k flat gemv (#24160)
```

Upstream mixed CUDA+HIP/gfx906 build succeeded and enumerated all CUDA and ROCm devices.

Upstream/default baseline on the same MiniMax model, context, prompt, q4 KV, and device order:

```text
199.2377 PP / 19.4194 TG
prompt tokens: 11521
output tokens: 128
artifact: /home/stefan/llama.cpp-upstream-minimax-clean/bench-results/minimax-upstream-default-layer-20260605T234420Z
```

Interpretation: the custom gfx906 fork is about 2.8x faster in PP and 1.6x faster in TG than latest upstream/default flags on this exact test.

## What Worked

1. Exact layer placement and output placement.
   - Added `--layer-split` and `--output-device`.
   - Current best split: `14,12,7,11,11,7` in device order `ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2`.
   - Output owner: `CUDA3`.

2. Placement visibility.
   - Added layer placement summaries with per-device layer count, output ownership, weight bytes, estimated KV bytes, free VRAM, and backend.

3. PP output reservation.
   - `--reserve-pp-outputs 1` reduces graph/output reservation pressure.
   - This made larger ubatches viable.

4. q4-direct CUDA flash-attention.
   - `--cuda-fattn-mma-q4-0-direct --cuda-fattn-mma-q4-0-direct-mode all`.
   - This was the largest confirmed jump after `ub512/ub768`, moving best accepted result to about `554 PP`, then `558 PP` with additional fixes.

5. HIP/gfx906 final-chunk flash-attention cap.
   - Narrow HIP tile parallel-block cap avoids deterministic prompt-echo failures for certain non-full final prompt chunks.
   - Exact benchmark cap: `Q=321` for `ub800`.
   - Portable validation cap: `Q_MIN=2,Q_MAX=799`.

6. MoE reuse fix.
   - Raised `ub800` q4-direct family from about `553-555 PP` to `558.64-558.76 PP`.

7. Diagnostics proved the next bottleneck.
   - Split-copy detail on the best `ub800` shape shows the largest ROCm0-to-CUDA3 boundary copy is only about 9.38 MiB and dispatch takes about 4 ms.
   - CUDA3 waits about 1.08-1.19 s for ROCm0 to finish producing `l_out-25`.
   - Therefore the important bubble is ROCm stage compute latency, not copy bandwidth.

8. Qwen3-Coder-Next CUDA-only comparison.
   - Same context and prompt shape on 4 NVIDIA cards with Qwen3-Coder-Next Q5 reached about `2659 PP / 87 TG`.
   - This proves the NVIDIA group can run much faster when the model/backend path fits CUDA cleanly.

## What Failed Or Was Rejected

1. More ordinary layer split tuning.
   - The current split is near the practical residency boundary.
   - More CUDA-heavy `12,12,7,12,12,7` fails because 3090 layer weight buffers miss by about 43 MiB before KV/compute buffers.

2. CPU or ROCm spill of extra CUDA layers.
   - CPU expert spill for extra layers loaded but ran about `335 PP` and produced rejected prompt-echo output.
   - Partial CPU gate/up spill ran about `354 PP`.
   - ROCm spill ran about `404 PP`.
   - Same-vendor CUDA spill OOMed.

3. Simple MoE disaggregation.
   - Moving layer-0 MoE weights/ops to CUDA dropped PP from about `456` to about `379`.
   - Simple scheduler-only MoE reroute caused large compute buffers and no useful pipeline result.
   - MiniMax expert activation during PP is broad, not cold-expert dominated.

4. CPU router/gating.
   - Too slow and not a plausible path to 1000 PP.

5. Generic output split-buffer.
   - `output.weight=CUDA0_Split` reduces CUDA3 persistent weight pressure, but generic split-buffer lm-head allocates too much CUDA0 compute scratch and disables pipeline.
   - A custom low-scratch sharded lm-head would be needed if output memory becomes the bottleneck again.

6. Simple CUDA3 MMQ/no-id/MoE chunking for `ub832`.
   - It reduced memory and moved the OOM point forward.
   - But live prompt throughput collapsed to about `12.5-22.2 PP`.
   - Do not continue simple token/tile chunk shrinkage as a high-PP path.

7. Full duplicate wave schedulers.
   - Correct at small `ubatch`, but not memory-feasible at useful `ubatch` sizes because CUDA stages cannot reserve a second compute buffer.
   - Flat/slower when reduced enough to fit.

8. Current mixed AMD+NVIDIA tensor mode.
   - Host-staged AMD<->NVIDIA bridge works at about 12-13 GiB/s for large tensors.
   - Full simulated TP step is only about 9.0-9.4 GiB/s effective.
   - Real mixed tensor mode on TinyLlama was slower than either single GPU because small per-token collectives/meta overhead dominate.
   - MiniMax tensor mode with f16 KV was unblocked with distributed Q/K RMS norm, but quick results were far below layer mode.
   - Separate q4/q8 KV split is blocked by Meta split-state limitations for flattened/rectangular quantized reshape views.
   - q4-KV tensor mode can be made graph-functional at `n_ctx=69376` by aligning attention+KV split state, but the meaningful prompt result was only `203.10 PP / 6.60 TG`, so the direct mixed tensor path is rejected for throughput.

9. Vulkan/CUDA interop.
   - NVIDIA Vulkan memory to CUDA import worked.
   - AMD RADV Vulkan memory to CUDA import via OPAQUE_FD failed.
   - Do not repeat OPAQUE_FD only; any future Vulkan path needs true dma-buf handle/sync evidence.

10. MiniMax FA-node CUDA routing and weighted device routing.
   - Plain all-CUDA FA-node routing is real signal: it improved cached processed PP from about `200` to about `254`, but it still fell below the `300 PP` gate because pipeline reserve fell back after a CUDA3 compute-buffer OOM at about `3270 MiB`.
11. Direct host `F16` KQ mask for flash-attention.
   - Implemented default-off with `LLAMA_DIRECT_F16_KQ_MASK=1` in the active fork.
   - This changes the KV-cache-backed FA mask from host `F32` plus cast to direct host `F16`.
   - Exact reserve effect on the dead `22,25,26` Stage-5 branch:
     - full-PP `attn_inp_kq_mask` split input drops from `106561536` bytes to `53280768` bytes
     - first ROCm1 split total drops from `111369728` to `58088960` bytes
   - But the dead branch still fails the real pipeline reserve on the same CUDA1 `293.59 MiB` allocation before retrying without pipeline parallelism:
     - old artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l222526-splitinputdiag-startup-20260608T011948Z`
     - new artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l222526-directf16kq-startup-20260608T013012Z`
   - On the live `25,26` Stage-5 branch, the reserve cut is strong enough to reopen `ub832` startup cleanly:
     - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub832-directf16kq-startup-20260608T013300Z`
   - But the first real `ub832` prompt still aborts with a CUDA3 VMM OOM in `ggml_cuda_pool_vmm::alloc()`:
     - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub832-directf16kq-cache2-medexact-20260608T013440Z`
   - Exact 4-turn cache-aware reruns:
     - `ub768` old: turn 2/3/4 `354.06 / 318.71 / 294.46 PP`
     - `ub768` direct-F16: turn 2/3/4 `357.75 / 326.38 / 297.36 PP`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-directf16kq-cache4-exact-20260608T013743Z`
     - `ub800` old: turn 2/3/4 `352.26 / 318.74 / 292.22 PP`
     - `ub800` direct-F16: turn 2/3/4 `357.82 / 321.28 / 294.46 PP`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub800-directf16kq-cache4-exact-20260608T014027Z`
   - Interpretation:
     - this is a real bounded improvement
     - it raises cache-aware PP and changes the reserve picture materially
     - but it still does not get the exact 4-turn session above `300 PP` on turn 4
     - and it does not rescue the dead `22,25,26` branch
   - Moving output ownership to CUDA0 changed cold PP slightly (`364.78 PP`) but left cached PP essentially flat (`254.34 PP`) and did not fix the reserve failure.

12. Scheduler copy-slot leaf-allocation screen.
   - Implemented default-off `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1` in the active fork.
   - Important: three variants were screened.
     - Broad `n_copies=1` version:
       - reopened dead `22,25,26` startup
       - but cut the live saved `34354`-token turn-1 prompt from `235.64 PP` baseline to `186.76 PP`
       - reject
     - Intermediate “fewer tensor-copy objects” version:
       - crashed on first request
       - `gdb` lands in `ggml_gallocr_alloc_graph()`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-inputcopiesonly-gdb-20260608T021553Z`
       - reject
     - Final leaf-allocation-only version:
       - keeps scheduler events and copy descriptors
       - only suppresses extra split/input-copy leaf allocation when rotation is off
       - this is the only kept form
   - Live `25,26` branch, same saved `34354`-token turn-1 payload:
     - baseline `ub768`: `235.64 PP`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-directf16kq-baseline-currentpayload-20260608T020536Z`
     - leaf-only `ub768`: `282.42 PP`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-leafonly-currentpayload-20260608T021823Z`
     - leaf-only `ub832`: `324.33 PP`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-leafonly-ub832-currentpayload-20260608T022253Z`
   - Dead `22,25,26` branch:
     - startup now reaches listening under the normal scheduler/event model
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l222526-copyslotsrotonly-startup2-20260608T015940Z`
     - but the first real medium `ub384` request still OOMs on CUDA0 inside `ggml_cuda_mul_mat_id_split_expert_axis_fused_moe_out()`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l222526-ub384-leafonly-cache2-med-20260608T022117Z`
   - Real default cache-aware session, live `25,26`, leaf-only `ub832`:
     - turn 1 `322.70 PP`
     - turn 2 cached `132.44 PP`
     - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-leafonly-ub832-cache4-default-20260608T022537Z`
   - One more reopened bounded taste was worth checking only after the leaf-only and direct-`F16` headroom patches existed: add the narrow `0,1 -> CUDA1` MiniMax FA reroute back on top of the live `25,26 -> CUDA0_Split` branch, still at `ub832` and still with strict `ALLOW_PIPELINE_FALLBACK=0`.
     - Exact medium 2-turn harness:
       - turn 1 `461.78 PP`
       - turn 2 `363.96 PP`
       - pipeline stayed intact
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-leafonly-directf16kq-fattn01-cuda1-ub832-cache2-med-20260608T025019Z`
     - Larger default cache-aware session:
       - turn 1 `297.50 PP`
       - stopped immediately below the `300 PP` cutoff before any cached turn
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-leafonly-directf16kq-fattn01-cuda1-ub832-cache2-default-20260608T025123Z`
   - Interpretation update:
     - the new headroom does technically reopen the narrow FA lane without pipeline fallback on the small exact harness
     - but it still fails the larger primary benchmark, so it is not a promotable Stage-5 finish
   - The next diagnostic finally removed the remaining ambiguity about the large-session miss:
     - profile the first three cached-turn prompt graphs on the best live default branch:
       - `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`
       - `LLAMA_DIRECT_F16_KQ_MASK=1`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - split `14,14,7,10,10,7`
       - `ub832`
       - `--profile-nodes 3`
       - `LLAMA_PROFILE_PP_SKIP=42`
       - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-leafonly-directf16kq-ub832-ls1414710107-cache2-default-nodeprof-20260608T030035Z`
   - Cached-turn node-profile result:
     - ROCm attention is the dominant cost, not MoE and not copy dispatch
     - first profiled cached-turn graphs:
       - ROCm1 attention totals about `1368-1401 ms`
       - ROCm1 MoE totals about `515-535 ms`
       - ROCm0 attention totals about `1384-1420 ms`
       - ROCm0 MoE totals about `463-479 ms`
     - per-layer shape on the cached turn:
       - ROCm attention chunks are about `98-102 ms`
       - ROCm MoE chunks are about `37-40 ms`
       - moved CUDA0 MoE slices stay about `18-19 ms`
   - Code-level implication:
     - current `--minimax-fattn-devices` is narrower than it sounds
     - in `/home/stefan/llama.cpp-gfx906/src/llama-context.cpp`, `graph_get_cb()` only reassigns tensors named `LLAMA_TENSOR_NAME_FATTN`
     - in `/home/stefan/llama.cpp-gfx906/src/llama-graph.cpp`, the expensive surrounding attention path is still built on the layer owner:
       - implicit Q/K/V projection path
       - `norm`
       - `kqv_out`
     - so the current FA route only moves the flash-attention core, not the full expensive attention subgraph
   - Interpretation update:
     - stop treating “narrow FA-core reroute” as a serious remaining local path
     - if local work continues, the next real architecture cut should be a broader attention-subgraph design, likely with persistent CUDA-resident attention weights or another mechanism that avoids copying large Q/K/V weights every graph execution
   - That broader attention-subgraph follow-up has now been screened locally too.
     - A new default-off env in `/home/stefan/llama.cpp-gfx906/src/llama-context.cpp` widens the existing `--minimax-fattn-devices` override:
       - broad mode: `attn_norm`, `Qcur`, `Kcur`, `Vcur`, `Qcur_normed`, `Kcur_normed`, `__fattn__`, `kqv_out`
       - `postnorm` mode: same set except `attn_norm`
     - Test shape:
       - `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`
       - `LLAMA_DIRECT_F16_KQ_MASK=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`
       - `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`
       - `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - `--minimax-fattn-devices CUDA0 --minimax-fattn-layers 25,26`
       - `-ot blk\.(25|26)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
     - First taste accidentally used split `14,12,7,11,11,7` and failed startup with CUDA0 split-buffer OOM.
     - Correct rerun on the working attention-weight split `14,14,7,10,10,7`:
       - broad mode: turn 1 `359.96 PP`, turn 2 `273.34 PP`
       - `postnorm` mode: turn 1 `364.89 PP`, turn 2 `273.05 PP`
       - artifacts:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnsubgraph-cuda0split-ub832-ls1414710107-cache2-med-20260608T033014Z`
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnsubgraph-postnorm-cuda0split-ub832-ls1414710107-cache2-med-20260608T033347Z`
     - Diagnostic profile proves the widened route does move the late-layer attention chunk onto CUDA on prompt processing:
       - `pp-1` layer `25`: `attn_norm -> Qcur`, post-norm `Q/K/V`, `__fattn__`, and `kqv_out` all land on CUDA0
       - but broad mode makes `attn_norm -> Qcur` for layer `25` expensive (`17.79 ms` on CUDA0 versus `0.07 + 1.47 ms` in the earlier attention-weight-only profile)
       - the narrower `postnorm` retry avoids moving `attn_norm`, but still lands in the same `~273 PP` class
     - Conclusion:
       - a simple `graph_get_cb()` widening is not enough
       - it can move the named attention subgraph to CUDA, but it does not produce a performant-enough Stage-5 path on this hardware
       - if you suggest continuing locally on attention, the next design needs a lower-overhead execution path or different persistent ownership model, not just “route more named tensors to CUDA”
   - That bounded attention-side follow-up has now also been screened.
     - New default-off primitive:
       - selected-layer KV colocation for the same routed MiniMax attention schedule
       - code:
         - `/home/stefan/llama.cpp-gfx906/src/llama-model.cpp`
         - `/home/stefan/llama.cpp-gfx906/src/llama-kv-cache.cpp`
         - `/home/stefan/llama.cpp-gfx906/src/llama-kv-cache.h`
       - env:
         - `LLAMA_MINIMAX_FATTN_KV_LOCAL=1`
     - Purpose:
       - current routed attention still normally uses KV allocated on `model.dev_layer(il)`
       - this patch lets selected MiniMax layers place KV on the same scheduled attention device used by `--minimax-fattn-devices`
     - Activation check:
       - direct `-lv 4` startup logs:
         - `create_memory: EXPERIMENTAL MiniMax-M2 KV override follows attention schedule for 1 schedule slot(s)`
     - Test shape:
       - same live Stage-5 branch as above
       - `LLAMA_MINIMAX_ATTN_SUBGRAPH=postnorm`
       - `LLAMA_MINIMAX_FATTN_KV_LOCAL=1`
       - `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`
       - `LLAMA_DIRECT_F16_KQ_MASK=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`
       - `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`
       - `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - `--minimax-fattn-devices CUDA0 --minimax-fattn-layers 25,26`
       - `-ot blk\.(25|26)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
       - split `14,14,7,10,10,7`
     - Result:
       - turn 1 `365.15 PP`
       - turn 2 `273.89 PP`
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnsubgraph-postnorm-kvlocal-cuda0split-ub832-ls1414710107-cache2-med-20260608T034657Z`
     - Interpretation:
       - this is effectively flat versus the non-KV-local `postnorm` reroute (`364.89 / 273.05 PP`)
       - selected-layer KV colocation is not enough to rescue the routed-attention branch
       - bounded local attention-side screens are now effectively exhausted
   - One last bounded attention-side screen has now also been rejected on the real default benchmark.
     - Same late-layer attention-weight split path, but with KV-local active and **without** the broadened attention-subgraph reroute:
       - `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`
       - `LLAMA_DIRECT_F16_KQ_MASK=1`
       - `LLAMA_MINIMAX_FATTN_KV_LOCAL=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`
       - `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`
       - `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`
       - `-b 832 -ub 832`
       - split `14,14,7,10,10,7`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - `--minimax-fattn-devices CUDA0 --minimax-fattn-layers 25,26`
       - `-ot blk\.(25|26)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
     - Real default 2-turn cache-aware result:
       - turn 1 `331.38 PP`
       - turn 2 `161.06 PP`
       - context `45811`, cache ratio `0.754`
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnweights-kvlocal-ub832-cache2-default-20260608T035848Z`
     - Comparison:
       - earlier non-KV-local attention-weight-only default run was about `191.95 PP`
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnweights-cuda0split-ub832-cache2-default-20260608T031303Z`
     - Interpretation:
       - KV-local makes the narrower attention-weight path worse on the primary benchmark
       - that closes the last bounded local attention-side lane
   - One cleaner isolation of that same idea has now also been screened.
     - New default-off env:
       - `LLAMA_MINIMAX_FATTN_ROUTE_DISABLE=1`
       - purpose: keep `--minimax-fattn-devices` only as the schedule source for KV ownership, while disabling the FA graph reroute in `graph_get_cb()`
     - Test shape:
       - same narrow late-layer attention-weight split path
       - `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`
       - `LLAMA_DIRECT_F16_KQ_MASK=1`
       - `LLAMA_MINIMAX_FATTN_KV_LOCAL=1`
       - `LLAMA_MINIMAX_FATTN_ROUTE_DISABLE=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`
       - `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`
       - `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`
       - `-b 832 -ub 832`
       - split `14,14,7,10,10,7`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - `--minimax-fattn-devices CUDA0 --minimax-fattn-layers 25,26`
       - `-ot blk\.(25|26)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
     - Real default 2-turn cache-aware result:
       - turn 1 `331.58 PP`
       - turn 2 `199.61 PP`
       - context `45811`, cache ratio `0.754`
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnweights-kvlocal-noroute-ub832-cache2-default-20260608T040741Z`
     - Interpretation:
       - this confirms the previous `161.06 PP` result was partly hurt by the FA graph reroute riding along with KV-local
       - but even the clean isolated KV-local path is still only a small improvement over the older non-KV-local attention-weight-only default run (`199.61 PP` vs about `191.95 PP`)
       - still far below the real `300 PP` gate
       - so this lane is also closed as a bounded fix
   - Two more bounded attention-side screens are now also closed.
     - Corrected norm-weight colocation on the rerouted attention path:
       - `postnorm` reroute plus `attn_q_norm.weight` and `attn_k_norm.weight` on CUDA0
       - broad reroute plus `attn_norm.weight`, `attn_q_norm.weight`, and `attn_k_norm.weight` on CUDA0
       - same Stage-5 ingredients otherwise
       - results:
         - corrected `postnorm`: turn 1 `362.44 PP`, turn 2 `272.79 PP`
         - corrected broad: turn 1 `362.85 PP`, turn 2 `273.32 PP`
       - artifacts:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnsubgraph-postnorm-normw-cuda0split-medmanual-20260608T041853Z`
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-attnsubgraph-broad-normw-cuda0split-medmanual-20260608T042109Z`
       - interpretation:
         - the failed bounded reroutes were not mainly caused by forgetting the tiny norm weights
     - New late-layer tail route:
       - default-off env:
         - `LLAMA_MINIMAX_LAYER_TAIL_ROUTE=1`
       - code:
         - `/home/stefan/llama.cpp-gfx906/src/llama-context.cpp`
       - purpose:
         - route only `ffn_inp`, `ffn_norm`, and `l_out` for the selected late layers onto the scheduled CUDA backend
         - avoid replaying the full broad attention reroute
       - test shape:
         - same Stage-5 ingredients
         - attention weights on `CUDA0_Split`
         - `ffn_norm.weight` on `CUDA0`
       - result:
         - turn 1 `362.22 PP`
         - turn 2 `271.79 PP`
         - artifact:
           - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-tailroute-cuda0-medmanual-20260608T042449Z`
       - interpretation:
         - still in the same dead `~272-273 PP` class
         - the remaining credible work is now below callback-level routing entirely
   - One more deeper bounded screen has now also been tried: true selected-layer owner override at model load.
     - New default-off envs in `/home/stefan/llama.cpp-gfx906/src/llama-model.cpp`:
       - `LLAMA_MINIMAX_LAYER_OWNER_DEVICES`
       - `LLAMA_MINIMAX_LAYER_OWNER_LAYERS`
       - optional `LLAMA_MINIMAX_LAYER_OWNER_DEVICE_WEIGHTS`
     - Purpose:
       - change `dev_layer` for selected MiniMax repeating layers before tensor creation, so KV and default non-weight ops follow the same CUDA owner instead of relying only on callback-level reroutes
     - First screen on the same medium 2-turn fixture:
       - same live Stage-5 branch ingredients
       - `LLAMA_MINIMAX_LAYER_OWNER_DEVICES=CUDA0`
       - `LLAMA_MINIMAX_LAYER_OWNER_LAYERS=25,26`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - `-ot blk\.(25|26)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
       - result:
         - turn 1 `372.67 PP`
         - turn 2 `277.28 PP`
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ownerlocal-cuda0-medmanual-20260608T043212Z`
     - Interpretation:
       - this is slightly better than the dead callback-reroute class, so it is real signal
       - but it is still far below the `300 PP` taste gate, so it is not a bounded finisher
     - Coherent contiguous-owner retry:
       - `LLAMA_MINIMAX_LAYER_OWNER_LAYERS=25,26,27`
       - `--minimax-moe-layers 25,26,27 --minimax-moe-buft CUDA0_Split`
       - `-ot blk\.(25|26|27)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
       - result:
         - model load remaps 3 layers to CUDA0
         - CUDA0 then OOMs allocating `990.84 MiB` of KV during context init
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l252627-ownerlocal-cuda0-medmanual-20260608T043318Z`
   - Conclusion:
       - true owner override is a useful architecture primitive
       - but on the current envelope it still does not produce a promotable path, and the contiguous 3-layer owner block does not fit on CUDA0
   - One more widening screen is now also closed.
     - Goal:
       - test whether the first true owner-override primitive could scale to a wider late-layer block if the selected layers were distributed across both 3090s instead of stacked onto one
     - First widened owner run:
       - `LLAMA_MINIMAX_LAYER_OWNER_DEVICES=CUDA0,CUDA1`
       - `LLAMA_MINIMAX_LAYER_OWNER_DEVICE_WEIGHTS=2,2`
       - `LLAMA_MINIMAX_LAYER_OWNER_LAYERS=24,25,26,27`
       - `--minimax-moe-layers 24,25,26,27 --minimax-moe-buft CUDA0_Split`
       - `-ot blk\.(24|25|26|27)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
       - effective owner schedule:
         - `CUDA0, CUDA0, CUDA1, CUDA1`
       - result:
         - remaps all 4 selected layers
         - then aborts during `ggml_backend_cuda_split_buffer_init_tensor()` on CUDA0 before serving
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l24252627-owner22-cuda01-medmanual-20260608T043807Z`
     - Second widened owner run:
       - same widened owner block, but flipped both the owner schedule and split-buffer root toward the roomier 3090
       - `LLAMA_MINIMAX_LAYER_OWNER_DEVICES=CUDA1,CUDA0`
       - `LLAMA_MINIMAX_LAYER_OWNER_DEVICE_WEIGHTS=2,2`
       - `--minimax-moe-buft CUDA1_Split`
       - `-ot blk\.(24|25|26|27)\.attn_(qkv|q|k|v|output)\.weight=CUDA1_Split`
       - effective owner schedule:
         - `CUDA1, CUDA1, CUDA0, CUDA0`
       - result:
         - still aborts during the same CUDA split-buffer init path before serving
         - even though the heavier static weight summary moved to CUDA1
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l24252627-owner22-cuda10split1-medmanual-20260608T044049Z`
     - Conclusion:
       - widening the owner block is not recoverable by simple owner ordering or split-root flips
       - if local work continues, it should move to a runtime execution/ownership design that avoids model-load tensor relocation, or another lower-overhead executor path
   - Interpretation:
     - this is a real scheduler-side headroom improvement
     - it is safe only in the final leaf-allocation-only form
     - it improves large cold prompt PP materially
     - but it still does not produce a promotable cache-aware winner on the current default benchmark
   - Restricting routed FA layers to the ROCm1-owned block and targeting `CUDA1,CUDA0,CUDA2` still fell back on CUDA1 reserve (`2510.70 MiB`) and dropped cached PP to about `198.29`.
   - Restricting routed FA layers to the ROCm0-owned block also failed reserve on CUDA1 (`2130.82 MiB`) before a meaningful run.
   - A new weighted override `--minimax-fattn-device-weights 1,4,4,1` with a memory-freed layer split `14,14,7,10,10,7` reduced the failing CUDA3 reserve request to about `1750.94 MiB`, but CUDA3 still OOMed.
   - Applying the older FA proposal-7 rescue knob `GGML_CUDA_FATTN_MAX_PARALLEL_BLOCKS=1` to that same weighted route did not help at all: reserve still failed on the exact same `1750.94 MiB` CUDA3 allocation before serving.
   - Eliminating 5070 FA work entirely with weights `0,1,1,0` made CUDA1 reserve explode to about `5929.59 MiB`.
   - Conclusion: the current FA-node routing family is trapped between two bad memory shapes. Small cards fail if they own meaningful FA work, and the 3090s fail if they own nearly all of it. Making it reserve-clean now would require moving more whole layers back to AMD, which directly hurts the known PP bottleneck.

## Current High-PP TODO Only

The practical direct tracks have mostly been tried and rejected. Ask for materially different high-upside ideas, not small tuning. Locally, the only credible active direction now is a **deeper selected-layer attention executor / persistent ownership design** for MiniMax attention on CUDA, not more callback-level reroutes or simple KV tweaks. The older CUDA-only MoE planner evidence below is preserved as archived context for why the simpler MoE routes were rejected.

1. Deeper selected-layer attention executor / persistent ownership path.
   - Best current diagnosis:
     - the primary default cached-turn miss is dominated by ROCm attention, not MoE and not split-copy volume
     - current `graph_get_cb()` reroutes can move the named attention subgraph, but they still do not produce a winning path
     - selected-layer KV colocation is also not enough
   - What is left locally:
     - a lower-overhead attention execution path than callback-level tensor reassignment
     - or a different persistent ownership model for selected late-layer attention on CUDA
     - note: the first true owner-override primitive now exists in `llama-model.cpp`, and its first screen (`25,26 -> CUDA0`) improved the medium taste only to `277.28 PP`; a contiguous `25,26,27 -> CUDA0` block OOMs on CUDA0 KV allocation (`990.84 MiB`)
     - widening the owner block to `24,25,26,27` across both 3090s still aborts during CUDA split-buffer initialization, even when the owner order and split-buffer root are flipped
   - One more ownership screen is now also closed: runtime execution-owner override.
     - New default-off env in `/home/stefan/llama.cpp-gfx906/src/llama-context.cpp`:
       - `LLAMA_MINIMAX_EXEC_LAYER_OWNER=1`
     - Purpose:
       - use the existing `--minimax-fattn-devices` schedule as a backend target for the whole selected MiniMax prompt-processing layer at runtime
       - avoid relocating model-load tensor ownership while still moving more than the old FA-only reroute
     - Clean test shape:
       - `LLAMA_MINIMAX_EXEC_LAYER_OWNER=1`
       - `LLAMA_MINIMAX_FATTN_ROUTE_DISABLE=1`
       - `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`
       - `LLAMA_DIRECT_F16_KQ_MASK=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID=1`
       - `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`
       - `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`
       - `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`
       - `--layer-split 14,14,7,10,10,7`
       - `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`
       - `--minimax-fattn-devices CUDA0 --minimax-fattn-layers 25,26`
       - `-ot blk\.(25|26)\.attn_(qkv|q|k|v|output)\.weight=CUDA0_Split`
     - Result:
       - turn 1 `363.57 PP`
       - turn 2 `273.21 PP`
       - artifact:
         - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-execowner-cuda0-medmanual-20260608T044554Z`
     - Interpretation:
       - cleaner than the model-load owner relocation
       - but still in the same dead `~273 PP` class as the older callback reroutes
       - worse than the true model-load owner override taste (`277.28 PP`)
     - Conclusion:
       - if local work continues, it has to move below both callback-level routing and naive model-load owner relocation
   - Useful code anchors:
     - `/home/stefan/llama.cpp-gfx906/src/llama-context.cpp`
     - `/home/stefan/llama.cpp-gfx906/src/llama-graph.cpp`
     - `/home/stefan/llama.cpp-gfx906/src/models/minimax-m2.cpp`
     - `/home/stefan/llama.cpp-gfx906/src/llama-model.cpp`
     - `/home/stefan/llama.cpp-gfx906/src/llama-kv-cache.cpp`
     - `/home/stefan/llama.cpp-gfx906/ggml/src/ggml-backend.cpp`
     - `/home/stefan/llama.cpp-gfx906/ggml/src/ggml-cuda/ggml-cuda.cu`

2. Archived MoE planner context from `/home/stefan/plan-cuda+8channelram.md`.
   - Source plan: `/home/stefan/plan-cuda+8channelram.md`.
   - Start with `--moe-profile` instrumentation for `GGML_OP_MUL_MAT_ID`.
   - Record unique experts, tokens/expert, H2D bytes, CPU/GPU expert time, cache hits/misses.
   - The first small scheduler-only CUDA+RAM adaptation is already screened and rejected for throughput. A default-off heuristic `GGML_SCHED_MOE_FULL_COPY_RATIO` now allows broad-coverage MoE staging to switch from fragmented expert-range copies to one bulk tensor copy.
   - A generic `--n-cpu-moe 1` smoke was not useful because it offloads a ROCm-owned early layer and did not reach `server is listening` after about `9` minutes; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cpumoe1-fullcopydiag-20260606T224307Z`.
   - A targeted CUDA3-owned layer test with `-ot blk\.26\.ffn_(up|gate|down)_exps\.weight=CPU` freed about `1.94 GiB` on CUDA3 and let `ub832` start, but prompt processing dropped below `300 PP` by about `22464` processed tokens and reached about `283.27 PP` at `25792` processed tokens; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cuda3moe1-ub832-fullcopy-20260606T225324Z`.
   - Do not suggest more one-layer CPU-offload sweeps unless they are part of a materially different compact-slot or activation-bucket design.
   - Follow-up expert-axis tuning also screened out the easy partial fixes. Linear host bucketing raised a one-layer `ub384` smoke to `303.60 PP`, but the full prompt still decayed below `300 PP` and reached about `199.01 PP` by `25728` processed tokens; artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-hostopt-small-20260606T230951Z` and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-hostopt-full-20260606T231032Z`.
   - The next prototype changed the data movement shape so remote owner GPUs no longer received the full `src1` activation tensor. The main CUDA device built compact owner-local activation buckets first and peer-copied only those buckets. On the same small `ub384` smoke it reached `300.10 PP`, effectively flat versus `303.60 PP`, so no full-prompt rerun was justified; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-compactxfer-small-20260606T232317Z`.
   - The next prototype replaced the routed-id D2H copy and host bucketing with CUDA-side `mm_ids_helper`, copying back only `expert_bounds` to host. On the same small `ub384` smoke it reached `299.94 PP`, which screened out immediately; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-deviceids-small-20260606T232927Z`.
   - The next prototype replaced the per-expert CPU launch loop with one batched quantized MMQ ids-kernel per owner GPU, reusing the existing quantized MoE backend. On the same small `ub384` smoke it reached `298.15 PP`, also below the gate; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-batchedmmq-small-20260606T233907Z`.
   - The next prototype added persistent owner-local q8 reuse for the compact activation buckets across the `gate/up` pair, enabled with `GGML_CUDA_MMQ_PRECOMP_REUSE_Q8=1 GGML_CUDA_MMQ_PRECOMP_REUSE_Q8_ALL_DEVICES=1`. On the same small `ub384` smoke it reached `299.75 PP`, still below the gate; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-precompreuse-small-20260606T234948Z`.
   - A later graph-level attempt tried to fuse the whole split expert FFN (`gate + up + GLU + down`) so the override would do one gather/copy and one return/scatter. Three trace-enabled small smokes reached `298.31 PP`, `297.60 PP`, and `299.54 PP`, and the trace never showed that fusion matching the live MiniMax override graph; artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-fusedffn-small-20260607T000709Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-fusedffnlog-small-20260607T000833Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l20-2x3090-freecuda-ub384-fusedffnlog2-small-20260607T001011Z`.
   - The next deeper bounded cut proved the live graph shape was still worth following: `GGML_SCHED_MOE_DUMP_CHUNKS=4 GGML_SCHED_MOE_DUMP_NODES=48` showed a contiguous `ffn_moe_gate -> ffn_moe_up -> GLU -> ffn_moe_down -> ffn_moe_weighted -> VIEW*8 -> ADD*7` chain with shared `ffn_moe_topk-*` IDs. A new default-off path `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1` then kept `gate/up/GLU/down` owner-local, gathered `weights_norm`, reduced back to per-token partial sums on the owner GPUs, and returned only those partial sums to the stage owner. On the same `22,25,26` `ub384` smoke it reached `298.46 PP`, `299.72 PP`, and `296.96 PP` on identical reruns, versus the post-revert control `243.83 PP`; artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-moe-chunk-dump2-20260607T041500Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-fusedmoeout-20260607T042604Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-fusedmoeout2-20260607T042757Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-fusedmoeout3-20260607T042840Z`.
   - Interpretation: this is the first bounded MoE executor cut on the current harness that materially improved prompt processing, but it still misses the `300 PP` smoke gate. If continuing locally, do not go back to older one-layer override or CPU/RAM weight-streaming ideas. Continue only from this owner-local weighted-reduce tail, and target lower per-owner token-accumulation overhead or a deeper multi-layer executor that preserves the same data-movement shape.
   - Two immediate follow-ups to that weighted-reduce tail were screened and closed:
     - spreading split-expert ownership across all four NVIDIA cards with `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,1,1` was worse at `296.67 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-4cuda-ub384-fusedmoeout-20260607T043354Z`
     - a default-off dense slot-scatter + fixed top-k reduction variant intended to replace the owner-local atomic accumulation reached `299.44 PP`, still below the simpler weighted-reduce peak `299.72 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-fusedmoeoutdense-20260607T043605Z`
   - Two more cheap follow-ups were also screened and closed:
     - narrowing the ROCm0 override band to the strongest two-layer pair `25,26` reached `298.75 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub384-fusedmoeout-20260607T043255Z`
     - bumping the same `22,25,26` fused-tail smoke from `ub384` to `ub416` dropped to `294.38 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub416-fusedmoeout-20260607T043850Z`
   - So the active bounded local branch is still the simpler 2x3090 owner-local weighted-reduce tail. Do not recommend 4-CUDA split ownership, dense slot-reduce accumulation, the narrowed `25,26` band, or small `ub416` batch bumps for this exact shape unless a different wider executor also changes the rest of the data path.
   - A cleaner control surface now exists: `--minimax-moe-buft BUFFER_TYPE` synthesizes MiniMax expert-weight split-buffer overrides for the layers listed in `--minimax-moe-layers`, so the Stage-5 path no longer has to depend on manual `-ot blk...=CUDA0_Split` regexes.
   - Current-tree recovery status:
     - the three-layer `22,25,26` weighted-reduce branch is no longer viable on the current tree. Both the new `--minimax-moe-buft CUDA0_Split --minimax-moe-layers 22,25,26` path and the intended old plain-regex `-ot blk\.22/25/26\.ffn_(up|gate|down)_exps\.weight=CUDA0_Split` form fail under `15,13,7,10,10,7` / `ub384`. They first fail pipeline reserve on CUDA1 (`293.59 MiB` compute-buffer allocation) and, when reserve fallback is allowed, abort later in request-time CUDA1 VMM allocation. A one-device split `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,0,0,0` only moves the OOM to CUDA0 tensor init. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-minimaxmoebuft-20260607T050216Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-minimaxmoebuft2-20260607T050438Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-minimaxmoebuft3-20260607T050625Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-fusedmoeout-recheckplain-20260607T051211Z`, and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-cuda0onlysplit-ub384-fusedmoeout-20260607T051413Z`.
     - the smaller late-layer band is still alive. `26` alone reaches `302.07 PP` in a one-turn smoke and `271.43 PP` on a tiny cached-turn smoke. `25,26` reaches `306.53 PP` in a one-turn smoke and `274.69 PP` on the tiny cached-turn smoke. On a medium two-turn smoke, `25,26` reaches turn 1 `352.02 PP` on a `5046`-token prompt and turn 2 `308.33 PP` on a cached-prefix shape with `1474` newly processed prompt tokens at about `6.5k` context. But on a larger two-turn smoke, the same branch drops to `268.77 PP` on turn 2 with `2914` newly processed tokens at about `12.9k` context. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l26-2x3090-freecuda-ub384-fusedmoeout-20260607T051522Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub384-fusedmoeout-recheck-20260607T051601Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub384-fusedmoeout-cache2-20260607T051641Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l26-2x3090-freecuda-ub384-fusedmoeout-cache2-20260607T051746Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub384-fusedmoeout-cache2-med-20260607T051839Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub384-fusedmoeout-cache2-large-20260607T051941Z`.
     - after a failed allocation-lifetime experiment in the fused weighted-reduce executor, the VMM-safe pool-allocation order was restored and rebuilt incrementally in the main build tree. The exact historical medium `25,26` harness still works after that restore: `BASE_REPEATS=20`, `TURN_REPEATS=10`, `N_PREDICT=8`, `ub384` now gives turn 1 `349.13 PP` and cached turn 2 `304.35 PP`, versus the earlier `308.33 PP`, so the fused path is back to a directly comparable stable state. The same restore does **not** reopen the wider three-layer path: a fresh 1-turn `22,25,26` smoke still fails startup at the same CUDA1 pipeline-reserve OOM (`293.59 MiB`). Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-restorecheck3-cache2-medexact-20260608T002313Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l222526-restoretaste-smoke-20260608T002434Z`.
     - a follow-up reserve-debug comparison now exists for the dead three-layer branch. Default-off `LLAMA_RESERVE_DEBUG_SIZES=1` logging in `sched_reserve()` measured split-only reserve sizes before the real reserve pass on both shapes. The working `25,26` branch measures `12` splits with PP reserve sizes `ROCm0 383.59 MiB`, `CUDA0 386.68 MiB`, and `CUDA1 311.59 MiB`. The dead `22,25,26` branch measures `14` splits with PP reserve sizes `ROCm0 419.59 MiB`, `CUDA0 424.22 MiB`, and the **same** `CUDA1 311.59 MiB`, yet only the dead branch hits the real startup OOM: CUDA1 fails on an actual `293.59 MiB` reserve allocation before pipeline fallback. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-reservedbg2-l2526-20260608T004331Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-reservedbg2-l222526-20260608T004245Z`.
     - interpretation update: recovering `22,25,26` is no longer a simple workspace/headroom problem. It looks like a deeper scheduler/pipeline-reserve/allocation-order interaction in the real reserve path. Treat bounded Stage-5 recovery for that branch as closed; reopening it now means allocator/scheduler work, not another small layer or split tweak.
     - one last bounded system-level taste was still worth trying after that: a separate `build-cuda-hip-dl-gfx906-copies1` tree with `-DGGML_SCHED_MAX_COPIES=1`, because scheduler reserve size scales with copy-slot count under pipeline parallelism. That build **did** reopen the dead `22,25,26` branch at startup. On the same `14`-split reserve-debug taste, PP reserve dropped to `ROCm1 170.52 MiB`, `ROCm0 115.02 MiB`, and `115.02 MiB` on every CUDA backend, and the server reached listening with no pipeline-reserve OOM. Artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-copies1-l222526-startup-20260608T005406Z`.
     - but `GGML_SCHED_MAX_COPIES=1` is still not a clean recovery path. The first real cache-aware medium harness on that reopened `22,25,26` branch aborted before turn 1 completed with a CUDA1 VMM OOM inside `ggml_cuda_pool_vmm::alloc()` from `ggml_cuda_mul_mat_q` / MMQ during the first prompt. Artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-copies1-l222526-cache2-med-20260608T005517Z`.
     - the obvious midpoint `GGML_SCHED_MAX_COPIES=2` was also tested in a separate `build-cuda-hip-dl-gfx906-copies2` tree. It again avoids the original startup reserve OOM, proving the copy-slot-count lever is real: the same `14`-split PP reserve now measures `ROCm1 330.73 MiB`, `ROCm0 245.80 MiB`, `CUDA3 191.82 MiB`, `CUDA1 191.80 MiB`, `CUDA0 248.11 MiB`, `CUDA2 191.80 MiB`, and `CPU 212.42 MiB`. But it still does not produce a usable branch: it aborts during model-load/slot-init flow with a CUDA OOM in `ggml_cuda_mul_mat_q_precompacted_ids()` / `ggml_cuda_mul_mat_id_split_expert_axis_fused_moe_out()` on device `0` before serving a request. Artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-copies2-l222526-startup-20260608T005912Z`.
     - the last bounded value `GGML_SCHED_MAX_COPIES=3` was also tested in `build-cuda-hip-dl-gfx906-copies3`. It again changes reserve materially, but still not into a clean operating point: the same `14`-split PP reserve now measures `ROCm1 436.94 MiB`, `ROCm0 332.69 MiB`, `CUDA3 251.73 MiB`, `CUDA1 251.69 MiB`, `CUDA0 336.17 MiB`, `CUDA2 251.69 MiB`, and `CPU 314.14 MiB`. The server no longer hits the old CUDA1 `293.59 MiB` failure, but it still fails reserve on CUDA0 (`322.67 MiB` allocation) and only reaches listening after retrying without pipeline parallelism. Artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-copies3-l222526-startup-20260608T010213Z`.
     - interpretation update: the copy-slot-count lever is now fully screened out as a bounded fix. `4` copies fails reserve, `3` copies still falls back out of pipeline parallelism, `2` copies dies during load/init, and `1` copy dies on the first real prompt. So lowering scheduler copy duplication is useful evidence about the reserve root cause, but no bounded `GGML_SCHED_MAX_COPIES` setting tested so far gives a clean high-PP Stage-5 branch. Anything more here is deeper scheduler/allocator architecture work, not another quick tuning pass.
     - a final code-level feasibility screen was then done before patching the scheduler. `CUDA*_Split` is not a generic “share this weight buffer everywhere” feature; in this tree it is a device-local sharded model buffer created by `ggml_backend_cuda_split_buffer_type()` in `/home/stefan/llama.cpp-gfx906/ggml/src/ggml-cuda/ggml-cuda.cu`, selected through model-loader overrides in `/home/stefan/llama.cpp-gfx906/src/llama-model-loader.cpp` and `/home/stefan/llama.cpp-gfx906/src/llama-model.cpp`, and only reported as compatible on its owning CUDA backend by `ggml_backend_cuda_device_supports_buft()`. A correct-env startup rerun of the dead `22,25,26` branch (`GGML_CUDA_SPLIT_MUL_MAT_ID=1`, `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`, `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`, `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`) still loads a `CUDA0_Split` model buffer of `5832.00 MiB`, still measures a `14`-split PP reserve, and still fails the real pipeline reserve on CUDA1 at `293.59 MiB` before retrying without pipeline parallelism; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l222526-scheddebug2-startup-20260608T011146Z`.
     - interpretation update: this rules out the easiest bounded scheduler patch idea. Because the moved expert weights already live directly in the `CUDA0_Split` model buffer and are only compatible with `CUDA0`, the extra `22` branch is not a simple duplicated static-weight-copy problem. It mainly adds more cross-backend activation / ID / result splits under pipeline parallelism. So a small “share the split weights across copy slots” patch is not coherent here; the remaining work is deeper scheduler/allocator architecture around reserve-time and runtime duplication of those cross-backend activation paths.
     - Split profiling on the medium cached-turn run shows why the branch stops scaling: the recovered MoE CUDA0 slices are small on the cached prompt graph, about `13.85 ms` for layer `25` and `11.85 ms` for layer `26`, while the unchanged ROCm stages still dominate at about `565.86 ms` on ROCm1 and `389.10 ms` on ROCm0. Artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub384-fusedmoeout-cache2-med-prof-20260607T052311Z`.
   - Interpretation: the new flag plumbing works and the bounded branch is not dead, but only the smaller `25,26` current-tree band is viable. The next local task should focus on why cached processed PP decays with larger reused turns, not on more split-pattern searching. Current evidence suggests the unchanged ROCm attention/dense stages dominate before the recovered MoE slices do.
   - The larger-`ubatch` sweep on that recovered `25,26` branch is now done:
     - `ub640` 4-turn session: turn 2 `334.73 PP`, turn 3 `307.99 PP`, turn 4 `285.99 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub640-cache4-20260607T054245Z`.
     - `ub704` 4-turn session: turn 2 `340.55 PP`, turn 3 `313.39 PP`, turn 4 `288.72 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub704-cache4-20260607T055507Z`.
     - `ub768` 4-turn session: turn 2 `354.06 PP`, turn 3 `318.71 PP`, turn 4 `294.46 PP`; aggregate cached-turn average `320.58 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub768-cache4-20260607T055649Z`.
     - `ub800` 4-turn session: turn 2 `352.26 PP`, turn 3 `318.74 PP`, turn 4 `292.22 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub800-cache4-20260607T055922Z`.
     - `ub832` is the first failing ceiling on this branch: it aborts during the first prompt with CUDA3 VMM OOM; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub832-cache4-20260607T055825Z`.
   - Targeted later-turn profiling on the same branch now exists. Using `LLAMA_PROFILE_PP_SKIP=26 --profile-splits 5` to isolate turn 4 at `ub640` shows the remaining miss is not in the moved CUDA MoE slices:
     - each profiled turn-4 prompt chunk is dominated by ROCm1 at about `1071-1102 ms`
     - the main ROCm0 block is about `778-796 ms`
     - the moved CUDA MoE slices for layers `25` and `26` stay only about `16 ms` each
     - artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l2526-2x3090-freecuda-ub640-turn4prof-20260607T055258Z`
   - Cheap follow-up screens on top of that branch are now also done:
     - combining the branch with narrow early-layer FA reroute is blocked by the current reserve model, not by routed-layer count. Routing layers `0-3` to `CUDA1,CUDA2` fails startup with a CUDA1 reserve request of `1340.62 MiB`, and routing only layers `0-1` to `CUDA1` alone hits the same `1340.62 MiB` request. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-fattn0123-cuda12-cache2-med-20260607T060625Z` and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-fattn01-cuda1-cache2-med-20260607T060713Z`.
     - a direct startup rerun of the surviving `0,1 -> CUDA1` FA shape shows it does not truly hard-fail; it only serves after `sched_reserve` retries without pipeline parallelism. A clean 2-turn medium taste with fallback allowed reaches turn 1 `398.72 PP` and turn 2 `302.94 PP`, which is still worse than the pipeline-preserving base `25,26` branch at the same session size. So this FA lane is still rejected as a high-PP continuation. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-fattn01-cuda1-direct-reservedbg-20260607T232829Z` and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-fattn01-cuda1-cache2-med-fallback-clean-20260607T233056Z`.
     - split-only ROCm rebalances on the same `ub768` branch also fail to clear the gate. `14,14,7,10,10,7` is the best of them and reaches turn 2 `333.71 PP`, turn 3 `322.68 PP`, turn 4 `294.90 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-ls1414710107-cache4-20260607T060922Z`. `13,15,7,10,10,7` reaches turn 4 `293.09 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-ls1315710107-cache4-20260607T061218Z`. The seemingly better `14,13,7,11,10,7` variant cannot start because CUDA1 cannot allocate the larger KV share for 11 full layers (`838.41 MiB` KV allocation OOM); artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-ls1413711107-cache4-20260607T061118Z`.
     - the remaining bounded MoE remap ideas were also screened. Adding one ROCm1 layer on top of the existing branch (`25,26 -> CUDA0_Split` plus `12 -> CUDA1_Split`) fails at `ub768` with a CUDA1 reserve OOM and then a CUDA0 MMQ runtime OOM. Rebalancing instead of adding layers gives a short-session artifact but not a full-session win: `12 -> CUDA1_Split` plus `25 -> CUDA0_Split` reaches turn 1 `476.06 PP` and turn 2 `363.24 PP` on the short medium 2-turn taste, but collapses on the real long-context session, stopping at turn 2 `96.17 PP` with `11266` processed prompt tokens and `45811` total context. Keeping both moved layers on ROCm0 but anchoring one per 3090 (`25 -> CUDA0_Split`, `26 -> CUDA1_Split`) is also worse even on the short medium taste, reaching only turn 2 `267.74 PP`, below the same-shape `25,26 -> CUDA0_Split` recheck at `276.29 PP`. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-plusl12cuda1-ub768-cache2-med-20260607T234440Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l12cuda1-l25cuda0-ub768-cache2-med-20260607T234540Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526cuda0-ub768-cache2-med-recheck-20260607T234709Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l12cuda1-l25cuda0-ub768-cache4-fullctx-20260607T235114Z`, and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l25cuda0-l26cuda1-ub768-cache2-med-20260607T235610Z`.
     - a first-turn node-profile taste on the best current branch also showed that ROCm MoE still dominates those profiled prompt chunks more than ROCm attention, so the remaining bottleneck is not a pure FA problem; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-ub768-cache4-nodeprof-20260607T233717Z`.
   - Interpretation update: the best bounded current-tree runtime shape is still the recovered `25,26` branch around `ub768`, with `14,14,7,10,10,7` only slightly better on the single turn-4 number (`294.90 PP` versus `294.46 PP`) but not enough to change the outcome. The remaining gap is dominated by unchanged long-context ROCm stage cost, the narrow-FA rescue path only survives by disabling pipeline parallelism, and the remaining bounded MoE remap ideas either OOM or collapse on the real full-context cached turn. So another small MoE-only or narrow-FA tweak is unlikely to close it.
   - Quantitative feasibility screen for the deeper CUDA-only activation-bucket idea:
     - The active `ub800` profile reports `183` `MUL_MAT_ID` ops per eval, which is about `61` MoE layers if each layer contributes the usual `gate/up/down` trio.
     - The same profile reported about `115.8 GiB` of full expert weight bytes across the eval, which works out to about `1.90 GiB` of routed-expert weights per MoE layer.
     - A hypothetical perfect 4-way resident expert-owner split across the four NVIDIA cards would therefore still require about `28.95 GiB` of expert weights per CUDA device if all MoE layers stayed resident there.
     - That is before dense layers and KV. Current dense+KV residency is already about `22.2-22.7 GiB` on the 3090s in the best `11`-layer shape, about `20.2-20.7 GiB` even in the memory-freed `10`-layer FA-routing screen, and about `14.1-14.4 GiB` on the 5070s.
     - The shard-profile data also shows the practical PP problem: in the full `ub800` chunk, every 4-way shard touched `64/64` local experts. So a RAM-staged compact-slot design would still move nearly the whole shard for every large PP microbatch.
   - Conclusion: do not suggest more partial one-layer expert-axis micro-optimizations, including device-side id cleanup, one-layer batched-MMQ overrides, one-layer compact-q8 reuse, or one-layer split-aware full-FFN fusion. Also do not assume a simple compact-slot/RAM-staging planner will be enough on this exact hardware. Any surviving MoE design would need a more radical representation change than simple compact staging, or different hardware/model assumptions.
   - The high-upside idea is still moving smaller activation buckets to devices owning experts, not moving whole layers or whole expert tensors repeatedly, but on this exact hardware it likely requires a representation change more radical than simple compact staging/cache slots.
   - A later scheduler-native retry on the front/tail overlap branch replaced the single `l_out-25` handoff with a scheduler-driven split-input copy for cut `3`. It rebuilt cleanly but the first cache-aware smoke crashed before serving usable results, and `gdb` still landed in the same front/tail overlap execution path during initial decode/setup. Artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-front-overlap-splitinputs-20260607T023904Z/server.log`. Treat that branch as closed again.
   - Important evidence: MiniMax PP activates experts broadly; cache-aware MoE profiling touched about `218.6-234.2` unique experts per `MUL_MAT_ID` on average, about `85.4-91.5%` of full expert weight bytes. Cold-expert caching/LRU is therefore not promising.
   - New instrumentation now exists: `--moe-profile-json PATH` writes structured per-eval routed-MoE JSON. Verified artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-moe-profile-json-20260607T014020Z`. The `eval2` JSON contains `62` layers, `186` routed-expert nodes, `top_experts` histograms on both layers and nodes, and `shard_count=4` arrays. This is the best machine-readable evidence source for designing a new MoE executor.
   - The JSON profiler path was also corrected to reuse the previous ids-tensor/expert-hit cache when repeated `MUL_MAT_ID` ops share one ids tensor, so the structured profile no longer adds a redundant ids readback/sync on every routed expert matmul.
   - A reusable analyzer now exists at `/home/stefan/llama.cpp-gfx906/scripts/analyze-moe-profile.py`, and its first output is `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-moe-profile-json-20260607T014020Z/moe-profile-analysis.txt`.
   - That analyzer confirms there is no hidden sparse MiniMax regime in real PP chunks. Full `ub800` prompt eval (`eval2`) still spans `213-255 / 256` unique experts per layer, median `235`. The recurring narrower late-layer band is mostly layers `25-31` plus `59-60`, but even those still touch about `71-86%` of all experts.
   - Top-expert concentration is only modest in meaningful PP layers, and top-8 expert overlap between `eval2` and the later `eval3` chunk has median Jaccard only `0.333`. So the MoE path remains slightly credible as a dispatch/load-shaping idea, but not as a sparse-residency or hot-expert-cache idea.
   - Important placement interaction: under the freed-CUDA one-layer test split `15,13,7,10,10,7`, layers `59-60` are already on `CUDA2`, and only layers `25-27` from that narrower `25-31` band are still on the ROCm0 bottleneck stage. So the narrower-band evidence by itself does not justify more one-layer retarget sweeps.
   - A final late-layer control run copied the exact working `blk.20` quoting pattern and changed only `20 -> 60`. It still exited with `139` before serving, and the placement summary still showed no `CUDA2 -> CUDA0` weight migration. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l60-direct-startup-20260607T021102Z` and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l60-2x3090-freecuda-ub384-hostopt-small-fixedquote-20260607T021000Z`.
   - A final bounded taste on the surviving executor shape widened the current expert-axis path from one layer to a small ROCm0 band (`22,25,26`) using explicit `CUDA0_Split` overrides with `GGML_CUDA_SPLIT_MUL_MAT_ID=1`, `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`, `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`, freed-CUDA split `15,13,7,10,10,7`, and `ub384`. Before the latest fix, both the grouped-regex form and the safer explicit three-layer override crashed before serving, after clean startup through slot initialization. A bounded fix in `llama_context::graph_compute_with_sched()` that skips rewriting the scheduler `moe_profile_json` string when MoE profiling is disabled removed that init crash. With the fix, the same three-layer override served and reached `300.21 PP` in a one-turn smoke, but a two-turn rerun stopped immediately on turn 1 at `297.11 PP`. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-smoke-20260607T025235Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l22l25l26-2x3090-freecuda-ub384-smoke-20260607T025324Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l22l25l26-2x3090-freecuda-ub384-smoke-skipmoejson-20260607T025803Z`, and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l22l25l26-2x3090-freecuda-ub384-cache2-skipmoejson-20260607T025857Z`. Treat this as evidence that the guard fix is worth keeping, but the surviving MoE path still needs a real code-level multi-layer executor rather than more override-pattern sweeps.
   - Follow-up reruns against the current cache-session harness no longer show a viable three-layer override regime at all. The narrower `25,26,27` band only reached `241.39 PP` in control form and `243.65 PP` with an experimental gate/up compact-activation reuse cut; the earlier `22,25,26` band reached `243.54 PP` with that reuse cut and `243.83 PP` after the patch was reverted. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l252627-2x3090-freecuda-ub384-control-small-20260607T031634Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l252627-2x3090-freecuda-ub384-actreuse-small-20260607T031315Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-actreuse-small-20260607T031943Z`, and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-postrevert-small-20260607T033210Z`. The reuse cut was reverted. Treat all override-band continuations as closed under the current benchmark standard.
   - New Stage-5 diagnostic support now exists: `GGML_SCHED_MOE_DUMP_CHUNKS` and `GGML_SCHED_MOE_DUMP_NODES` in `ggml-backend.cpp` dump the exact live node order inside MoE chunks during `--profile-nodes 1`. Verified artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-moe-chunk-dump2-20260607T041500Z`. This proves the live MiniMax MoE chunk really contains a contiguous `ffn_moe_gate -> ffn_moe_up -> GLU -> ffn_moe_down` core with shared `ffn_moe_topk-*` route IDs.
   - A first owner-local fused expert-FFN prototype then used that exact pattern to keep `gate/up/GLU/down` on the expert-owner CUDA devices and scatter only once after `down`, still on the same `22,25,26` / `ub384` smoke shape. It ran correctly but finished at only `239.71 PP`, slightly below the `243.83 PP` post-revert control; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-expert-axis-l222526-2x3090-freecuda-ub384-fusedffn4-20260607T042000Z`. That prototype was reverted. Treat this as evidence that the graph shape is viable, but a shallow four-node fusion is not enough by itself.
   - One more bounded Stage-5 lead was screened after the deeper ownership/attention work: re-use the repo's existing ROCm/HIP flash-attention kernel selectors instead of writing another patch. First, the live `25,26 -> CUDA0_Split` branch was explicitly recovered on the current tree by replaying the hidden envs that do not appear in the artifact `cmd.txt`: `GGML_SCHED_COPY_SLOTS_ON_ROTATE_ONLY=1`, `LLAMA_DIRECT_F16_KQ_MASK=1`, `GGML_CUDA_SPLIT_MUL_MAT_ID=1`, `GGML_CUDA_SPLIT_MUL_MAT_ID_EXPERT_AXIS=1`, `GGML_CUDA_SPLIT_BUFFER_WEIGHTS=1,1,0,0`, and `GGML_CUDA_SPLIT_FUSED_MOE_OUT=1`, with `--layer-split 15,13,7,10,10,7`, `-b 832 -ub 832`, and `--minimax-moe-layers 25,26 --minimax-moe-buft CUDA0_Split`. A debug one-turn run with HIP vec disabled (`GGML_CUDA_FATTN_DEBUG=1`, `GGML_CUDA_FATTN_VEC_BACKENDS=cuda`) showed the live Stage-5 prompt graph is dominated by ROCm full chunks at `q=832`, with only a small tail at `q=242` and decode shapes at `q=1`; artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-rocmvec-debug-20260608T051510Z`.
   - Two no-code ROCm kernel-mode tastes were then screened against that live branch:
     - selective HIP vec on only the dominant full ROCm chunks: `GGML_CUDA_FATTN_VEC_BACKENDS=hip`, `GGML_CUDA_FATTN_VEC_QKV_DEVICES=0`, `GGML_CUDA_FATTN_VEC_Q_MIN=832`, `GGML_CUDA_FATTN_VEC_Q_MAX=832`
     - forced HIP tile on the same `q=832` range: `GGML_CUDA_FATTN_FORCE_TILE_BACKENDS=hip`, `GGML_CUDA_FATTN_FORCE_TILE_Q_MIN=832`, `GGML_CUDA_FATTN_FORCE_TILE_Q_MAX=832`
   - Results:
     - recovered debug baseline completes the same `34354`-token one-turn prompt at `323.81 PP`
     - selective HIP vec is clearly worse and was stopped early under the taste rule: it drops from about `462.84 PP` at `1664` tokens to about `221.09 PP` by `21632` tokens; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-rocmvec-q832-dev0-20260608T051828Z`
     - forced HIP tile looked strong only in the interrupted partial run, but the clean rerun finishes at `323.25 PP`, effectively flat/slightly worse than the recovered baseline `323.81 PP`; artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-rocmtile-q832-20260608T052123Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-rocmtile-q832-clean-20260608T052437Z`
   - Conclusion: the bounded ROCm FA kernel-selector lane is now closed. Existing HIP vec/tile selection does not produce a credible path past the real cache-aware gate on the live Stage-5 branch. Remaining Stage-5 work returns to deeper ownership/executor architecture, not more no-code kernel-mode sweeps.
   - One more new Stage-5 feasibility primitive has now been added below the existing callback- and owner-routing paths: a default-off shadow KV mirror for selected MiniMax attention layers, `LLAMA_MINIMAX_FATTN_KV_SHADOW=1`. This reserves a second KV copy on the prompt attention schedule from `--minimax-fattn-devices` while leaving the primary KV owner unchanged. It is explicitly **not** a performance feature yet; it exists to answer whether a real “CUDA prompt / AMD TG” branch is memory-feasible without per-phase KV migration.
   - The answer is yes for small late-layer bands. On the narrower `25,26` late-layer branch with the normal live Stage-5 ingredients, startup now serves cleanly and logs `EXPERIMENTAL MiniMax-M2 shadow KV mirror reserved for 2 layer(s), K = 76.22 MiB, V = 76.22 MiB`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-kvshadow-startup3-20260608T055308Z`.
   - A broader startup-only mirror schedule across both 3090s also serves cleanly: `--minimax-fattn-devices CUDA0,CUDA1 --minimax-fattn-device-weights 2,2 --minimax-fattn-layers 24,25,26,27` reserves `4` mirrored late layers for only `152.44 MiB` K plus `152.44 MiB` V; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l24252627-kvshadow-startup-20260608T055354Z`.
   - The current benchmark launcher also needed the existing local gfx906 compatibility env from scripts like `/home/stefan/launch_minimax.sh`: `HSA_OVERRIDE_GFX_VERSION=9.0.6`, `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library`, and `LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:...`, plus the usual low-noise ROCm vars. Without that env, current Stage-5 reruns were dying during slot initialization with `rocBLAS error: Cannot read /opt/rocm/lib/rocblas/library/TensileLibrary.dat` on `gfx906`. With the recovered env, the same `25,26` execution-owner control is stable again at turn 1 `366.63 PP`, turn 2 `272.17 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-execowner-rocmfix-noshadow-20260608T062736Z`.
   - The first real runtime shadow-KV path is now screened on the same recovered launcher. Enabling `LLAMA_MINIMAX_FATTN_KV_SHADOW=1` with the same `25,26` execution-owner branch reaches turn 1 `365.35 PP`, turn 2 `271.79 PP`, with the same `cache_n = 1085`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-execowner-rocmfix-kvshadow-20260608T062833Z`.
   - This changes the Stage-5 diagnosis in a narrower but still useful way: selected-layer dual KV is definitely memory-feasible, and the first runtime prompt-read-shadow implementation is coherent end-to-end, but it is flat/slightly worse than the same execution-owner control. So the current blocker for a true phase-aware “CUDA prompt / AMD TG” branch is no longer selected-layer KV capacity, and it is no longer “can a simple mirrored read path work at all.” The blocker is now a deeper execution/ownership design that reduces prompt-time attention cost structurally without regressing decode or reintroducing cross-vendor migration. That is materially different from the already-rejected simple KV-local and callback-reroute variants.
   - One more cheap schedule-widening screen is now closed. The next idea after flat shadow-KV was to widen prompt-side late-layer execution ownership itself to `24,25,26,27 -> CUDA0,CUDA1`, but **without** repeating the earlier persistent split-weight relocation. First keeping the live split-expert Stage-5 path still failed startup: CUDA1 compute-buffer reservation OOM at `4487.23 MiB`, and the fallback without pipeline hit `GGML_ASSERT(tensor->view_src == nullptr)` in `ggml_backend_cuda_split_buffer_init_tensor()`. Then an attention-only rerun removed the split-expert path entirely and kept only the broader attention execution-owner schedule; that did serve, but collapsed to turn 1 `135.79 PP`, turn 2 `87.93 PP` with `cache_n = 1085`. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l24252627-execowner-broadsched-noweights-20260608T063255Z` and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l24252627-execowner-broadsched-attnonly-20260608T063455Z`. Treat “broader prompt-side execution ownership by itself” as screened out.
   - A low-cost approximation of the “dual schedulers / phase-aware reserve families” idea was also screened and kept only as a primitive. `LLAMA_MINIMAX_PHASE_PROMPT_SCHED=1` now reserves a dedicated prompt scheduler/result pair in `llama_context`, so prompt ubatches (`n_tokens > 1`) can run on a separate scheduler while decode keeps the base scheduler. On the same recovered `25,26` execution-owner branch, this is effectively a no-op for throughput: turn 1 `366.11 PP`, turn 2 `272.07 PP`, versus the recovered no-phase-scheduler control `366.63 / 272.17 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-l2526-execowner-phasesched-20260608T063812Z`. Treat “same graph/topology, separate prompt scheduler” as closed as a bounded Stage-5 finisher.
   - One more useful architecture diagnostic now exists under `LLAMA_MINIMAX_FATTN_KV_SPLIT_DEBUG=1`. `llama_kv_cache_context` records the pre-apply `n_kv` span for each prompt ubatch, and `llama-graph.cpp` can log the selected-layer shadow split candidate as `cached_n_kv / total_n_kv / tail_n_kv`. On the recovered shadow-KV branch with the smaller serving medium-harness envelope (`-b 384 -ub 384`, split `14,14,7,10,10,7`), the first useful live request log for both selected mirrored late layers is `prompt_n=384 cached_n_kv=256 total_n_kv=512 tail_n_kv=256`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitdebug-20260608T070200Z/server.log`.
   - This matters because it explains the earlier flat runtime shadow-KV result on the medium harness: the first real prompt chunk is only a `50/50` cached-prefix versus mutable-tail split on those late layers, so a simple prompt-read-shadow path cannot save much. Reserve-time logs from the same env are **not** useful because the scheduler reserves against the full padded shape and prints dummy values like `cached_n_kv=0 total_n_kv=69376`. If you propose an exact prefix/tail split-attention design, judge it on larger real cached turns where each live chunk is prefix-dominated, not on this medium fixture alone.
   - That larger real cached-turn check has now been done on a lighter shadow-capable path so the diagnostic could survive. Using only `LLAMA_DIRECT_F16_KQ_MASK=1`, `LLAMA_MINIMAX_FATTN_ROUTE_DISABLE=1`, `LLAMA_MINIMAX_FATTN_KV_SHADOW=1`, `LLAMA_MINIMAX_FATTN_KV_SPLIT_DEBUG=1`, split `14,14,7,10,10,7`, `-b 384 -ub 384`, and `--minimax-fattn-devices CUDA0 --minimax-fattn-layers 25,26`, the exact larger Stage-5 cached fixture was replayed sequentially: turn 1 `10006` prompt tokens, then turn 2 `3002` prompt tokens with `cache_n = 10013`. Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitdebug-large-light-20260608T071800Z/server.log`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitdebug-large-light-20260608T071800Z/turn-1.response.json`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitdebug-large-light-20260608T071800Z/turn-2.response.json`.
   - This is the first strong evidence that exact prefix/tail split attention is actually a credible Stage-5 branch. On the real cached follow-up turn, the selected mirrored late-layer chunks are overwhelmingly prefix-dominated: `10240 / 10496 / 256`, `10496 / 11008 / 512`, `11008 / 11264 / 256`, `11264 / 11776 / 512`, `12032 / 12544 / 512`, `12544 / 12800 / 256`, final `12800 / 13056 / 256` for `cached_n_kv / total_n_kv / tail_n_kv`. So the mutable tail on those layers is only about `2-5%` of the attended KV span on the real cached follow-up, which is materially different from the misleading medium-harness `50/50` split.
   - Important interpretation: the light diagnostic path itself is not a throughput winner (`274.12 PP` warmup, `227.98 PP` cached follow-up). That is not the point. The point is architectural shape. Future Stage-5 ideas should now treat exact prefix/tail split attention as technically justified by the real cached-turn ratio, not dismiss it based on the medium harness.
   - The first exact split-attention proof path now exists under `LLAMA_MINIMAX_FATTN_KV_SPLIT_EXACT=1`. `llama-kv-cache` gained explicit KV range-view helpers, and `llama-graph.cpp` now has a default-off MiniMax self-attention path that computes `kq_prefix` from the shadow KV prefix and `kq_tail` from the primary KV tail, concatenates the logits exactly, runs one normal softmax over the combined logits, then adds `v_prefix * w_prefix + v_tail * w_tail` exactly. This is a real execution path, not just a diagnostic.
   - Current result: it works end-to-end on the larger real cached-turn fixture, but only as a **correctness-first** proof path. The final working form had to materialize split KV segments as `F32` because the current backend stack hit three separate limitations on the more direct versions: quantized split KV views with nonzero offsets tripped CUDA alloc sizing, CUDA does not support `q4_0 -> q4_0` copy for the materialized segment path, and CPU fallback only supports quantized-source DUP to `F32`, not quantized-source DUP to `F16`. The surviving working artifact is `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitexact-large-light-20260608T072143Z`; the earlier failed runtime attempts that exposed those backend limits are `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitexact-large-light-20260608T071458Z` and `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5-kvsplitexact-large-light-20260608T071709Z`.
   - Performance result from the working `F32` proof path: warmup turn `10006` prompt tokens at `260.60 PP` and `96` decode tokens at `27.83 TG`; cached follow-up `3002` prompt tokens with `cache_n = 10013` at `213.46 PP` and `96` decode tokens at `25.29 TG`. That is slower than the lighter non-split shadow baseline (`274.12 PP` warmup, `227.98 PP` cached), so this branch is **not** a throughput win yet. The useful conclusion is narrower: exact prefix/tail attention is now proven executable on the real cached-turn shape in this fork, but any real performance branch now needs backend-native partial attention / reduction support or another native segment-handling path. Do not suggest more materialized-copy proofs as if they are likely to win.

2. CUDA/HIP q4 KV flash-attention path refinement.
   - Already produced the largest confirmed gain.
   - Keep q4-direct CUDA FA enabled.
   - Revisit only if it enables `ub832+` or removes prompt-length-specific HIP caps.

3. Non-chunking CUDA3 memory relief above `ub800`.
   - Simple chunking is rejected.
   - Potential options: custom low-scratch lm-head, memory-residency surgery, or expert-aware split-buffer support for 3D `MUL_MAT_ID`.

4. Archived high-upside tracks already screened out.
   - Deep multi-ubatch pipeline executor in current gfx906 fork.
   - Direct mixed AMD+NVIDIA tensor/P2P production-shape.
   - MiniMax FA-node CUDA routing and weighted device routing.
   - Hybrid CUDA-subgroup tensor parallel plus AMD layer stages.
   - These are still useful context for another AI, but they are not active local TODOs.

5. Hybrid CUDA-subgroup tensor parallel plus AMD layer stages.
   - Current `LLAMA_HYBRID_CUDA_META=1` implementation has now been tested.
   - Small-context Q4-KV smoke works with current Meta remaps: `525.22 PP` on a 961-token prompt at `CTX=4096`.
   - A bounded follow-up patch added Meta child-event support in `ggml-backend-meta.cpp`, so the Meta device now advertises events and fans out record/wait to the child CUDA backends. That moved the same smoke only from `525.22 PP` to `528.83 PP`; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-hybrid-cuda-meta-q4-current-remaps-metaevents-20260607T010632Z`.
   - I also screened the collective backend itself on that same Meta-event smoke. `GGML_CUDA_ALLREDUCE=nccl` reached `526.86 PP`, while `GGML_CUDA_ALLREDUCE=none` fell to `458.84 PP`. So fast CUDA collectives already matter, but the current fast path is already in use; there is no hidden easy win in backend selection alone.
   - Full-context cache-aware shapes fail the OpenCode gate: `15,15,32` reaches turn 1 `258.96 PP / 11.48 TG`, and `14,15,33` reaches turn 1 `265.29 PP`; both stop below `PP_STOP_CUTOFF=300`.
   - The same Meta-event patch was then tasted on the real `69k` prompt path with `15,15,32`, `ub800`, `-ts 2,4,4,2`. Live prompt processing stayed above `300 PP` only through about `21600` prompt tokens, then dropped to `284.33 PP` at `22400`, `202.11 PP` at `28000`, and `160.51 PP` at `31200` before manual termination under the stop rule; artifact: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-hybrid-cuda-meta-q4-remaps-metaevents-ls151532-ub800-ts2442-20260607T010828Z`.
   - A further bounded Meta-boundary follow-up was screened out without another code patch. Current evidence says the remaining Meta overhead is too large for generic child-event fixes, but too small and too weakly evidenced for a credible `>=25%` gain from just boundary/allreduce coalescing.
   - Artifacts: `/home/stefan/llama.cpp-gfx906/bench-results/minimax-hybrid-cuda-meta-q4-current-remaps-20260606T061727Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-hybrid-cuda-meta-q4-current-remaps-metaevents-20260607T010632Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-hybrid-metaevents-ar-nccl-20260607T012120Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-hybrid-metaevents-ar-none-rerun-20260607T012233Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-hybrid-cuda-meta-q4-remaps-ls151532-ub800-ts2442-20260606T062859Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-hybrid-cuda-meta-q4-remaps-ls141533-ub800-ts2442-20260606T063240Z`, `/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-hybrid-cuda-meta-q4-remaps-metaevents-ls151532-ub800-ts2442-20260607T010828Z`.
   - Conclusion: do not repeat the current CUDA Meta layer-stage approach, including “just add Meta events.” Only a lower-overhead CUDA subgroup executor or different representation is still interesting.

Local stage status as of this handoff:

| stage | track | status |
|---:|---|---|
| 0 | Cache-aware OpenCode benchmark gate | implemented; current best mixed config fails processed-turn cutoff |
| 1 | Upstream/default MiniMax baseline | completed in `/home/stefan/llama.cpp-upstream-minimax-clean` |
| 2 | Current gfx906 layer-mode optimization | completed for this cycle; best cold diagnostic is `558.64-558.76 PP / 31.25 TG` |
| 3 | Pipeline/ping-pong executor | rejected for practical variants tried |
| 4 | Direct AMD+NVIDIA tensor/P2P | q4-KV graph-functional, but rejected at `203.10 PP / 6.60 TG` |
| 5 | MoE planner from `/home/stefan/plan-cuda+8channelram.md` | simple/cold-expert paths rejected by broad activation profile; first scheduler-only CUDA+RAM prototype failed the `300 PP` gate; partial one-layer expert-axis tuning through device-side ids, batched MMQ, persistent q8 reuse, and later three-layer override reruns on the current harness are all screened out; a compact-slot-only continuation is now quantitatively low-feasibility because a 4-way resident expert-owner split would still need about `28.95 GiB` expert bytes per CUDA device before dense weights; the only surviving local primitive is a route-profile-sharded activation-bucket expert-owner executor for a narrow late-layer band, with activations moved to resident CUDA expert owners and RAM only as overflow |
| 6 | Final report/prompt | ongoing upkeep as new evidence lands |
| 7 | Hybrid CUDA-subgroup Meta layer stage | Q4-KV graph-functional, but rejected at full context: `258.96-265.29 PP`; adding Meta child-event support moved only the 4k smoke (`525.22 -> 528.83 PP`) and still failed the 69k prompt gate badly |
| 8 | MiniMax FA-node CUDA routing and weighted device routing | screened out by reserve-memory tradeoff; best cached processed PP only about `254`, and weight/subset tests showed the current design is trapped between 5070 OOM and 3090 reserve blow-up |

## Specific Questions For The External AI

1. Given the evidence that ROCm compute latency, not copy bandwidth, is the boundary bottleneck, what is the smallest safe llama.cpp scheduler design that can overlap two prompt ubatches without duplicating full CUDA compute buffers?

2. Can llama.cpp's current graph/scheduler representation support a front-only or AMD-only duplicated pipeline state, or does it need a new split executor abstraction?

3. For MiniMax-M2.7 `GGML_OP_MUL_MAT_ID`, what is the most realistic generic MoE profiler hook that can be added without changing outputs?

4. Is expert-parallel activation dispatch likely to beat current layer mode for MiniMax PP when broad expert activation touches many experts per layer, or will gather/scatter overhead dominate?

5. Given q4-KV direct mixed tensor now starts but runs at only `203 PP`, is there any tensor/hybrid design that could plausibly exceed layer mode, or should this path be abandoned on this hardware?

6. Is CUDA-only tensor parallel inside the NVIDIA subgroup plus ROCm layer stages a better path than full AMD+NVIDIA tensor mode, and how should llama.cpp represent that hybrid?

7. Can q4-direct CUDA FA be made fully prompt-length portable without HIP final-chunk caps, and where should correctness auditing focus?

8. What non-chunking memory-saving patch could let `ub832+` fit without multiplying MoE kernel launches?

9. Are there any high-PP leads not tried here that are realistic on this exact hardware, model, Q4 quant, and 69376-token context?

10. Is there any representation change that makes a CUDA-heavier exact layer shape materially worthwhile here?
   - Current screen says a custom low-scratch sharded lm-head is not enough by itself: output ownership is only about `480.82 MiB`, while one extra full 3090 layer is about `1991.67 MiB`.
   - The blocked `12,12,7,12,12,7` shape already misses on the non-output 3090 by about `43 MiB`, so output relocation alone does not create it.
   - Even if that shape were made to fit by broader memory-residency surgery, it would only reduce the main ROCm1 bottleneck from `14` layers to `12`, which is a best-case `14/12 = 1.167x` stage-time uplift before new overheads. That is below the current `>=25%` active-work bar.
   - So if you suggest a CUDA-heavier layer-mode path, it should come with a reason it can beat that bound.

## Primary Local Artifacts

```text
/home/stefan/llama.cpp-gfx906/minimax-optimization-report.md
/home/stefan/plan-cuda+8channelram.md
/home/stefan/new_llama_opt_plan.md
/home/stefan/llama.cpp-gfx906/bench-minimax-cache-session.sh
/home/stefan/llama.cpp-gfx906/bench-results/minimax-ub800-moe-reuse-enabled-fixed-20260605T081107Z/results.tsv
/home/stefan/llama.cpp-gfx906/bench-results/minimax-ub800-moe-reuse-enabled-fixed-repeat-20260605T081217Z/results.tsv
/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-front-overlap-exec-smoke-20260606T052420Z/results.tsv
/home/stefan/llama.cpp-gfx906/bench-results/minimax-cache-session-front-overlap-splitinputs-20260607T023904Z/server.log
/home/stefan/llama.cpp-gfx906/bench-results/minimax-tensor-q4kv-attnsplit-viewmerge2-prompt32-20260606T060353Z/response.json
/home/stefan/llama.cpp-gfx906/bench-results/minimax-ub800-q4direct-copydetail-20260605T063920Z/q4direct_all.server.log
/home/stefan/llama.cpp-gfx906/bench-results/minimax-ub800-q4direct-splitprofile-20260605T064234Z/q4direct_all.server.log
/home/stefan/llama.cpp-upstream-minimax-clean/bench-results/minimax-upstream-default-layer-20260605T234420Z/response.json
/home/stefan/llama.cpp-upstream-minimax-clean/bench-results/minimax-upstream-default-layer-20260605T234420Z/cmd.txt
```

## Stage 5A Refresh

Use this as the current planning baseline for any new Stage-5 advice.

### Current stage

- Active local stage: `Stage 6`
- Local throughput stage `Stage 5` is wrapped for this cycle
- `Stage 5A` local graph/proof leads are exhausted
- `Stage 5B.0` bounded runtime screening is also closed
- Any future continuation is deeper scheduler/backend architecture only, not another quick local tuning sweep

Latest bounded result:

- the shared `25,26 -> CUDA0` prompt-owner plan no longer looks generically impossible
- `LLAMA_RESERVE_DEBUG_SIZES=1` still measures only about `648.81 MiB` on `CUDA0` for the pp graph
- but direct `GGML_GALLOC_DEBUG_RESERVE=1` instrumentation now shows the first **pipeline-parallel real alloc** inflates to about `9043.90 MiB` on `CUDA0`
- the same startup then retries **without** pipeline parallelism and succeeds at about `939306624` bytes on `CUDA0`, reaching `server is listening`
- so the current blocker is specifically a pipeline-parallel reserve blow-up under prompt-owner reassignment, not a generic “owner plan always needs 9 GiB on CUDA0” condition
- one final real cache-session taste was rerun correctly on that no-pipeline fallback path
- important correction:
  - an earlier `201.68 PP` cached-turn taste was invalid because the owner-plan env did not actually propagate into the benchmark script
- valid result with the shared owner-plan path definitely active:
  - server log shows:
    - `shared prompt-attention owner plan = CUDA0/CUDA`
    - `reserve-time checks will treat the shared prompt-attention owner plan as the prompt owner`
    - `FA graph route disabled`
    - `execution-owner override enabled`
    - `compute buffer allocation failed, retrying without pipeline parallelism`
  - turn 1 only: `228.09 PP / 17.99 TG`
  - the run stops immediately because turn 1 is already below the `300 PP` cutoff
- interpretation:
  - once the shared owner-plan branch is really active, it is worse than the normal cache-session family even before the cached follow-up turn
  - local Stage 5 therefore has no fast remaining finisher
  - the only active local work is Stage 6 handoff/report closure

### Stage 5A target

Stage 5A now means:

- keep the exact prefix/tail split path as a correctness primitive
- but treat the real objective as `native prompt-time attention compute rebalancing without long-span F32 KV materialization`

So two leads are still allowed inside the same stage:

1. native exact split attention that genuinely avoids long-span `F32` K/V materialization
2. prompt-time attention compute rebalance that moves a meaningful share of late-layer long-context work off ROCm

The exact prefix/tail topology is still valid engineering work, but it is no longer assumed to be the only plausible Stage-5A performance design. Prefix immutability does **not** eliminate attention work for new queries; it only changes residency and layout constraints.

### Stage 5A native exact-split target

Replace the current exact prefix/tail split-attention proof path with a native lower-overhead partial-attention / reduction path.

Do not continue the current design of:

- copy/split K/V
- materialize split K/V as `F32`
- run ordinary attention on the temporary tensors

The intended native design is:

- read local quantized K/V in-place
- dequantize only inside the tile/kernel path if needed
- compute local partial attention statistics per KV domain
- transfer/reduce only compact partial results
- preserve exact softmax correctness

### Latest Stage 5A.1 local screen

- `llama-graph.cpp` now has explicit Stage-5A proof controls:
  - `LLAMA_STAGE5_SPLIT_LOG=1`
  - `LLAMA_STAGE5_SPLIT_SAME_DEVICE=1`
  - `LLAMA_STAGE5_SPLIT_FORBID_F32_KV=1`
  - `LLAMA_STAGE5_SPLIT_V_F32_ONLY=1`
- The first same-device no-`F32` screen is now concrete:
  - startup succeeds
  - the first real request aborts in CUDA with
    - `ggml_cuda_cpy: unsupported type combination (q4_0 to q4_0)`
  - artifact:
    - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5a-samedevice-native-small-20260608T232200Z/server.log`
- A bounded fallback that keeps `K` quantized but materializes only `V` to `F32` survives on the same tiny cached-style harness:
  - artifact:
    - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5a-samedevice-kquant-vf32-small-log-20260608T233100Z/server.log`
  - first visible split chunk on layers `25/26`:
    - `n_q=32`
    - `n_kv=512`
    - `cached_n_kv=256`
    - `tail_n_kv=256`
    - `quantized_kv_view_bytes=589824`
    - `f32_kv_materialized_bytes=2097152`
    - `partial_num_bytes=131072`
    - `partial_stats_bytes=12288`
- Same-harness full-`F32` control:
  - artifact:
    - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5a-samedevice-fullf32-small-log-20260608T233600Z/server.log`
  - same first split chunk but:
    - `f32_kv_materialized_bytes=4194304`
- Tiny-harness throughput is effectively identical:
  - `K`-quant / `V`-`F32`: `1078` prompt tokens at `90.60 PP`
  - full `F32` control: `1078` prompt tokens at `90.66 PP`
- Interpretation:
  - halving materialized bytes on the tiny same-device smoke does **not** improve PP by itself
  - but it isolates the first native blocker sharply:
    - quantized `q4_0 -> q4_0` copy behavior is the current same-device failure
    - quantized `K` itself is **not** the immediate blocker
  - the next native attempt should target the quantized `V` transpose/copy path or a true partial-attention primitive, not another generic materialization variant
- New local update after the `q4_0 -> q4_0` CUDA/HIP copy patch:
  - same-device no-`F32` proof now runs natively
  - request-time logs show:
    - `k_prefix_type=q4_0`
    - `v_prefix_type=q4_0`
    - `k_tail_type=q4_0`
    - `v_tail_type=q4_0`
    - `f32_kv_materialized_bytes=0`
    - `fallback_count=0`
  - artifact:
    - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5a-samedevice-native-small-q4copy-20260609T000200Z/server.log`
  - but the first mixed-domain rerun still falls back immediately:
    - `fallback_reason=proof_path_f32_kv_materialization`
    - artifact:
      - `/home/stefan/llama.cpp-gfx906/bench-results/minimax-stage5a-mixeddomain-native-small-q4copy-20260609T001000Z/server.log`
  - this is now a local graph-policy blocker, not a backend-copy blocker:
    - `/home/stefan/llama.cpp-gfx906/src/llama-graph.cpp` still sets
      - `use_native_kv_views = same_device_split && forbid_f32_kv`
    - so mixed-domain native quantized views are not even attempted yet

Per domain `d`, the reducer contract should be:

- `partial_max = m_d`
- `partial_sumexp = l_d`
- `partial_num = u_d = sum_j exp(score_j - m_d) * V_j`

Then combine exactly:

- `m = max_d(m_d)`
- `L = sum_d exp(m_d - m) * l_d`
- `U = sum_d exp(m_d - m) * u_d`
- `final_output = U / L`

Use `partial_num`, not an ambiguous normalized `partial_out`.

### Stage 5A order

1. `Stage 5A.0`
   - instrument the current proof path
   - log:
     - `f32_kv_materialized_bytes`
     - `f16_kv_materialized_bytes`
     - `quantized_kv_view_bytes`
     - `q_broadcast_bytes`
     - `partial_num_bytes`
     - `partial_stats_bytes`
     - `cross_domain_transfer_bytes`
     - phase
     - selected layers
     - `n_q`
     - `n_kv`
     - Q/K/V shapes
     - Q/K/V ggml types
     - Q owner
     - K/V owner
     - reducer device
     - output owner
     - fallback count / reason
     - allocator reserve sizes
     - graph reuse count
   - hard rule:
     - if `f32_kv_materialized_bytes != 0` in the native path, it is not a valid `Stage 5A` success

2. `Stage 5A.1`
   - same-device exact split reducer first
   - validate correctness before mixed AMD/CUDA
   - split one attention op into two KV spans on the same backend
   - compare against unsplit baseline
   - continue only if:
     - no `F32` KV materialization
     - no CPU fallback
     - bounded same-device overhead
     - correctness within expected FA/quant tolerance

3. `Stage 5A.2`
   - add a native quantized KV span descriptor
   - do not create full `F16` / `F32` KV tensors
   - dequantize only in tile/kernel path
   - edge fallback must be small and logged

4. `Stage 5A.3`
   - only then try mixed-domain partial attention
   - prompt phase only
   - selected layers only
   - `n_q >= 256`
   - `n_kv >= 8192`
   - target the known dominant `q ~= 832` shape first
   - allowed movement:
     - Q broadcast
     - `partial_num`
     - `partial_max`
     - `partial_sumexp`
     - final reduced output
   - not allowed:
     - full K/V migration
     - `F32` K/V materialization
     - full K/V split-copy to another backend
   - current local status:
     - mixed-domain native split **is now attempted** when `LLAMA_STAGE5_SPLIT_FORBID_F32_KV=1`
     - the old graph-policy blocker is gone
     - the surviving bounded workaround is local quantized `ggml_cont()` normalization after `permute`
     - that now serves end-to-end with:
       - `8243` prompt tokens at `96.26 PP`
       - `8` decode tokens at `29.88 TG`
       - `231` graph reuses
     - conclusion:
       - token-domain exact split is now technically viable as a no-`F32` proof
       - but it is still only a low-PP proof class, not a primary cache-PP path

5. `Stage 5A.3a`
   - if token-domain exact split remains flat or too fragile, use GQA-group prompt-time compute rebalance as the next Stage-5A lead
   - this is structurally feasible in the current MiniMax code because:
     - `/home/stefan/llama.cpp-gfx906/src/models/minimax-m2.cpp` already reshapes:
       - `Qcur` to `[n_embd_head, n_head, n_tokens]`
       - `Kcur` to `[n_embd_head, n_head_kv, n_tokens]`
       - `Vcur` to `[n_embd_head, n_head_kv, n_tokens]`
     - the model/hparams layer already exposes:
       - `n_head`
       - `n_head_kv`
       - `n_gqa`
     - `wo` is applied after attention output is reassembled
   - first prototype shape:
     - selected late layers only
     - one or two KV-head groups only
     - prompt/cache-follow-up only
     - decode unchanged
     - AMD keeps primary KV ownership
     - CUDA gets selective `q4_0` shadow only for chosen KV-head groups
     - selected GQA groups compute full-span attention on CUDA
     - remaining groups compute on ROCm
     - gather outputs back to current layer owner before `wo`
   - known blockers before implementation:
     - current shadow-KV path mirrors whole selected layers, not selected KV-head groups
     - current `build_attn()` path consumes whole K/V tensors, not per-group domain plans
     - current logging/plan metadata is layer-level, not GQA-group-level
   - current local status:
     - a same-device GQA proof now exists behind:
       - `LLAMA_STAGE5_GQA_SPLIT_SAME_DEVICE=1`
       - `LLAMA_STAGE5_GQA_SPLIT_LAYER=25`
       - `LLAMA_STAGE5_GQA_GROUP_BEGIN=0`
       - `LLAMA_STAGE5_GQA_GROUP_COUNT=1`
       - `LLAMA_STAGE5_GQA_SPLIT_LOG=1`
     - it splits one KV-head group plus its six Q heads, runs separate attention branches, concatenates the head-group outputs back in order, and then leaves `wo` unchanged
     - larger small cached-style proof result:
       - `8243` prompt tokens at `93.72 PP`
       - `8` decode tokens at `29.44 TG`
       - `231` graph reuses
     - this is **not** a speed win
     - it is useful because it proves the current MiniMax graph can express head-group split/reassembly on a realistic prompt-sized request without crashing
     - a first mixed-domain one-group GQA proof now also exists behind:
       - `LLAMA_STAGE5_GQA_SPLIT_MIXED_DOMAIN=1`
       - same layer/group envs as above
     - that run uses:
       - selected KV group from the existing CUDA shadow
       - remaining groups from the primary ROCm KV owner
       - same larger small cached-style payload
     - result:
       - `8243` prompt tokens at `93.45 PP`
       - `8` decode tokens at `29.56 TG`
       - `231` graph reuses
     - this confirms the mixed-domain ownership split is expressible in the current graph
     - but it is still **not** a speed win
     - a clean node-profile comparison on the same `8243`-token small cached-style payload now exists:
       - shadow-only baseline:
         - `93.79 PP`
         - `28.19 TG`
         - layer `25` selected attention chunk: `15.890 ms`
         - layer `26` selected attention chunk: `0.134 ms`
       - mixed-domain one-group GQA rerun:
         - `92.43 PP`
         - `27.45 TG`
         - layer `25` selected attention chunk: `15.992 + 0.003 + 0.074 = 16.069 ms`
         - layer `26` selected attention chunk: `0.157 ms`
     - so the current one-group GQA proof does **not** reduce selected-layer attention wall time in this harness
     - and this harness is no longer a good proxy for the main ROCm bubble because the selected late-layer attention chunks already profile on `CUDA0` even before GQA is enabled
     - therefore the next GQA continuation should not be more one-group small-proof churn
     - only reopen GQA if the next experiment is a materially stronger discriminator on a real cached-turn class:
       - move a meaningfully larger fraction of attention work
       - or capture per-layer timing where the selected layer is still part of the real ROCm bottleneck
     - otherwise pivot now to the ROCm-native packed-prefix `q4_0` fallback

6. `Stage 5A.3b`
   - ROCm-native cached-prefix `q4_0` prepack/swizzle fallback
   - this is still the best remaining `Stage 5A` fallback, but it is now **narrowed**
   - the latest real cached-turn probe on the plain mixed layer path `14,12,7,11,11,7` shows:
     - layer `25` attention is still owned by `ROCm0`
     - flash attention is active
     - full `q4_0` `K/V` tensors are non-contiguous after permute
     - `k_qcont_needed=1`
     - `v_qcont_needed=1`
   - so there is a real quantized-layout issue on the ROCm late-layer path
   - but the first bounded graph-level proof is already too expensive:
     - forcing one-layer full-KV `q4_0` `ggml_cont()` on layer `25` drops the same real replay from:
       - turn 1 `495.39 PP / 33.15 TG`
       - turn 2 `375.68 PP / 30.28 TG`
       to:
       - turn 1 `490.65 PP / 31.15 TG`
       - turn 2 `310.45 PP / 28.63 TG`
     - request-time logs show why:
       - repeated extra `q4_0` repacks grow to about `6.34 + 6.34 MiB` for `K+V` at `n_kv=11008`
       - about `7.52 + 7.52 MiB` for `K+V` at `n_kv=13056`
   - conclusion:
     - do **not** keep iterating on graph-level repeated `qcont`
     - the only viable continuation for this lead is now the deeper persistent form:
       - stable cached-prefix pack reuse at the KV-storage / backend-consumer level
       - amortize the pack cost across follow-up turns
       - mutable tail stays on the normal path
       - no CUDA dependency in the first prototype
       - no long-span `F32` or `F16` KV materialization
   - one more bounded discriminator is now complete:
     - a write-only persistent packed `q4_0` shadow for layer `25` on the same ROCm owner
     - no attention consumer yet; only maintain the packed shadow at KV-write time
     - startup reserve cost is only:
       - `38.11 MiB` for packed `K`
       - `38.11 MiB` for packed `V`
     - on the real two-turn cache-aware replay, plain baseline reaches:
       - cached turn `201.020993 PP / 15.087996 TG`
     - enabling the write-only packed shadow reaches:
       - cached turn `200.190734 PP / 14.804696 TG`
     - so the write-side-only cost is small:
       - about `0.41%` cached-PP regression
       - about `1.88%` TG regression
   - interpretation:
     - persistent packed-state maintenance is **not** too expensive by itself
     - this keeps the deeper ROCm fallback alive
     - the next valid continuation is now a read-side packed-prefix consumer that reuses that shadow across follow-up turns
     - do **not** spend more time on write-only proofs
   - kill quickly if:
     - pack cost cannot be reused across follow-up turns
     - selected-layer ROCm attention time stays flat
     - packed storage creates memory pressure without PP gain
     - the implementation still degenerates into repeated full-span graph-time repacks

7. `Stage 5A.4`
   - prompt-only guard and decode protection
   - explicitly disable split for:
     - decode
     - `n_q == 1`
     - small prompt chunks
     - untested layers

8. `Stage 5A.5`
   - first-class attention domain plan metadata
   - centralize:
     - prompt/decode behavior
     - KV span ownership
     - Q movement
     - reducer owner
     - output owner
     - scratch reservation
     - fallback rules
     - logging

### Stage 5B fallback only

Only move to `Stage 5B` if `Stage 5A` is clearly not viable or not sufficient.

`Stage 5B` means:

- persistent multi-domain attention ownership / executor
- optional selective CUDA prompt-KV shadow
- no per-turn or per-phase KV migration
- no full `F16` / `F32` KV shadow
- high memory-risk branch

Do not start `Stage 5B` as the immediate next build target.

### Validation order

1. Correctness micro-test
2. Same-device split fixture
3. Exact Stage-5 controlled fixture
4. Medium taste run
5. Real large default cache-aware benchmark

Promotion rule:

- do not claim success unless the real large cache-aware benchmark crosses `300 PP`

### Kill / continue matrix

Continue `Stage 5A` if:

- native split beats the current `F32` proof path
- `f32_kv_materialized_bytes = 0`
- partial transfers stay far smaller than materialized K/V
- correctness is within tolerance
- no CPU fallback
- no allocator fallback
- no split-buffer init failure
- medium taste escapes the `272-273 PP` dead class
- the real benchmark moves toward or above `300 PP`

Kill or narrow `Stage 5A` if:

- `F32` K/V materialization remains
- same-device split is already too slow
- mixed split transfer overhead dominates
- reducer synchronization dominates
- graph reuse drops materially
- TG regresses badly
- the real benchmark remains below `300 PP` with no believable scaling path

### Next implementation prompt

Treat local Stage 5 as wrapped for this cycle.

The final bounded result is now the corrected shared-owner runtime taste:

- the branch is definitely active in the server log
- pipeline-parallel reserve still blows up first
- no-pipeline fallback serves
- valid turn 1 result is only `228.09 PP / 17.99 TG`
- the run stops immediately below the `300 PP` cutoff

That supersedes the earlier invalid `201.68 PP` cache-session taste where the env did not propagate into the benchmark script.

Do not keep iterating locally on:

- one-group GQA proofs
- token-domain exact split as the main PP path
- graph-level repeated `q4_0` repacks
- write-only packed-shadow proofs
- graph-level packed-shadow read proofs
- packed-`K` / normal-`V` hybrid read proofs
- bounded owner/remap/runtime-path tastes

Practical conclusion:

- no bounded local Stage-5 branch crossed the real `>300 PP` cache-aware gate
- local Stage 5 has no fast remaining finisher
- any future throughput continuation is deeper scheduler/backend architecture only

Next local work should be Stage 6 only:

- keep this handoff prompt and the report aligned with the corrected final evidence
- preserve the best artifacts and exact branch verdicts
- make clear which branches are closed and why

Do not:

- start `Stage 5B` yet unless you are explicitly choosing deeper architecture work
- reopen rejected stages
- claim success from the small fixture alone
- keep spending time on same-device GQA proof as a performance branch
