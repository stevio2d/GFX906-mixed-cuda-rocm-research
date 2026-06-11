# GFX906 Mixed CUDA+ROCm Research

Curated handoff repo for the MiniMax-M2.7 mixed NVIDIA + AMD llama.cpp research cycle.

This repo is not a clean product branch. It is a research package for continuing the work with:

- the final report
- the release note
- the external-AI handoff prompt
- the exact best commands
- selected benchmark artifacts
- the modified source snapshot
- the benchmark and helper scripts used during the investigation

## Tested Hardware

The research in this repo was run on a mixed host with:

- `2x NVIDIA RTX 3090` linked with `NVLink`
- `2x NVIDIA RTX 5070 Ti`
- `2x AMD MI50 32 GiB`

## Current Status

- Local throughput work is wrapped for this cycle.
- No bounded local branch crossed the real `>300 PP` cache-aware gate.
- The remaining future work is deeper scheduler/backend architecture only.

Authoritative summary docs:

- [CONTINUATION_LEADS.md](CONTINUATION_LEADS.md)
- [RELEASE_NOTES.md](RELEASE_NOTES.md)
- [docs/minimax-optimization-report.md](docs/minimax-optimization-report.md)
- [docs/minimax-external-ai-prompt.md](docs/minimax-external-ai-prompt.md)
- [docs/plan-cuda+8channelram.md](docs/plan-cuda+8channelram.md)
- [docs/q8-kv-validation-2026-06-09.md](docs/q8-kv-validation-2026-06-09.md)

## Best Commands

### Highest cold PP

Result:

- `558.7603 PP / 31.2581 TG`
- repeat: `558.6422 PP / 31.2530 TG`

What this used:

- static `-sm layer` multi-GPU placement across all `6` GPUs
- device order: `ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2`
- manual layer split: `14,12,7,11,11,7`
- coarse placement outcome:
  - `26` layers on the `2x MI50`
  - `36` layers on the `4x NVIDIA`
  - final output on `CUDA3`
- `q4_0` KV cache
- CUDA-side `q4_0` direct MMA path enabled

Important:

- this best cold result did **not** come from a successful fine-grained “move low-active pieces to MI50 and hot pieces to NVIDIA” runtime design
- the winning cold shape was primarily a strong static layer placement plus CUDA-side attention/MMA settings

Artifacts:

- [artifacts/best-cold-pp](artifacts/best-cold-pp)
- [artifacts/best-cold-pp-repeat](artifacts/best-cold-pp-repeat)

Runnable reference:

- [commands/best-cold-pp.sh](commands/best-cold-pp.sh)
- required ROCm env helper: [commands/env-rocm-gfx906.sh](commands/env-rocm-gfx906.sh)

Full command line:

```bash
/home/stefan/llama.cpp-gfx906/build-cuda-hip-dl-gfx906/bin/llama-server \
  -m /home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf \
  -c 69376 -ngl 999 -sm layer \
  -dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2 \
  -fa on -np 1 -fit off --numa distribute --no-warmup --jinja \
  -t 24 -tb 48 -lv 3 \
  -ctk q4_0 -ctv q4_0 \
  -b 800 -ub 800 \
  --placement-policy memory \
  --device-speed 0.55,0.55,1.0,1.0,1.0,1.0 \
  --cuda-fattn-no-stream-k \
  --cuda-fattn-vec-qkv-headroom-mib 0 \
  --reserve-pp-outputs 1 \
  --layer-split 14,12,7,11,11,7 \
  --output-device CUDA3 \
  --cuda-fattn-mma-q4-0-direct \
  --cuda-fattn-mma-q4-0-direct-mode all \
  --host 127.0.0.1 \
  --port 21523
```

Required FA env for the best cold artifact:

```bash
export GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS=hip
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=321
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=321
export GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=1
```

### Highest real cache-session PP

Best real cache-session cached follow-up result found in the main benchmark family:

- cached turn `processed_pp = 255.317006`
- cached turn `effective_context_pp = 1036.021386`
- cached turn `TG = 3.405396`

Important:

- this is the highest cached follow-up PP that was recorded in the main cache-session runs
- it still failed the project promotion gate of `>300 PP`

What this used:

- same coarse static `-sm layer` split as the best cold command:
  - `ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2`
  - `--layer-split 14,12,7,11,11,7`
- `q4_0` KV cache
- prompt cache enabled
- output on `CUDA3`
- extra cache-session optimization on top:
  - `--minimax-fattn-devices CUDA3,CUDA1,CUDA0,CUDA2`

Interpretation:

- the best cache-session family still relied mainly on static layer ownership
- the extra cache-session twist was to prefer NVIDIA devices for the selected MiniMax prompt-time flash-attention route
- it was **not** a successful general dynamic scheduler that classified “low active” versus “high active” blocks at runtime

Artifacts:

- [artifacts/best-cache-session](artifacts/best-cache-session)
- comparison baseline: [artifacts/cache-baseline](artifacts/cache-baseline)

Runnable reference:

- [commands/best-cache-session.sh](commands/best-cache-session.sh)
- required ROCm env helper: [commands/env-rocm-gfx906.sh](commands/env-rocm-gfx906.sh)

Full command line:

```bash
/home/stefan/llama.cpp-gfx906/build-cuda-hip-dl-gfx906/bin/llama-server \
  -m /home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf \
  -c 69376 -ngl 999 -sm layer \
  -dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2 \
  -fa on -np 1 -fit off --numa distribute --no-warmup --jinja \
  -t 24 -tb 48 -lv 3 \
  -ctk q4_0 -ctv q4_0 \
  -b 800 -ub 800 \
  --placement-policy memory \
  --device-speed 0.55,0.55,1.0,1.0,1.0,1.0 \
  --cuda-fattn-no-stream-k \
  --cuda-fattn-vec-qkv-headroom-mib 0 \
  --reserve-pp-outputs 1 \
  --layer-split 14,12,7,11,11,7 \
  --output-device CUDA3 \
  --cuda-fattn-mma-q4-0-direct \
  --cuda-fattn-mma-q4-0-direct-mode all \
  --cache-prompt \
  --cache-ram 32768 \
  --minimax-fattn-devices CUDA3,CUDA1,CUDA0,CUDA2 \
  --host 127.0.0.1 \
  --port 21880
```

### Q8 KV validated server

Follow-up Q8 KV validation was completed after the main handoff.

Key result:

- full context `69376`
- `q8_0` KV
- tested cold run at `309.394175 PP / 30.657956 TG`
- tested two-turn cache smoke with real reuse

What this used:

- stable full-context Q8 KV server shape
- balanced placement rather than the old fixed `14,12,7,11,11,7` split
- no `--minimax-fattn-devices` override in the validated stable run

Reason:

- the Q8 server tolerated `b200/ub200`
- the more aggressive Q8 runs and the Q8 FA-reroute variant did not survive request-time memory pressure

Reference docs and scripts:

- [docs/q8-kv-validation-2026-06-09.md](docs/q8-kv-validation-2026-06-09.md)
- [scripts/run-minimax-q8-opencode.sh](scripts/run-minimax-q8-opencode.sh)
- [scripts/bench-minimax-q8-best-cold.sh](scripts/bench-minimax-q8-best-cold.sh)

Full command line:

```bash
/home/stefan/llama.cpp-gfx906/build-cuda-hip-dl-gfx906/bin/llama-server \
  -m /home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf \
  -c 69376 -ngl 999 -sm layer \
  -dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2 \
  -fa on -np 1 -fit off --numa distribute --no-warmup --jinja \
  -t 24 -tb 48 -lv 3 \
  -ctk q8_0 -ctv q8_0 \
  -b 200 -ub 200 \
  --placement-policy balanced \
  --device-speed 0.55,0.55,1.0,1.0,1.0,1.0 \
  --cuda-fattn-no-stream-k \
  --cuda-fattn-vec-qkv-headroom-mib 0 \
  --reserve-pp-outputs 1 \
  --cuda-fattn-mma-q4-0-direct \
  --cuda-fattn-mma-q4-0-direct-mode all \
  --output-device CUDA3 \
  --cache-prompt \
  --cache-ram 32768 \
  --host 0.0.0.0 \
  --port 21940
```

Portable FA env used for the validated Q8 server:

```bash
export GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS=hip
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=2
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=799
export GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=1
```

## What Was Actually Optimized

The short answer for anyone comparing results:

- the best published runs were mainly **static layer placement** wins
- not a successful general runtime policy that automatically moved “low active” parts to MI50 and “hot” parts to NVIDIA

What was consistently true in the strongest runs:

- MI50s carried a large early-layer share
- NVIDIA carried the larger late-layer share plus output
- CUDA-side attention/MMA options mattered
- cache-session runs sometimes benefited from CUDA-preferred prompt-time attention routing

What did **not** become a validated winner:

- narrow callback reroutes
- token-domain exact split as a PP path
- one-group GQA as a PP path
- graph-level packed-prefix read/write proofs
- broader prompt-owner remaps

### Best controlled exact Stage 5 fixture

Result:

- turn 2: `408.65 PP`
- turn 3: `386.81 PP`
- turn 4: `365.85 PP`

Artifact:

- [artifacts/best-exact-stage5-fixture](artifacts/best-exact-stage5-fixture)

This fixture is diagnostic only. It is not the promotion metric.

## Final Failing Branch

The last bounded Stage 5 runtime branch that was still worth checking was the shared prompt-owner path.

Valid final result:

- turn 1 only: `228.09 PP / 17.99 TG`
- stopped below the `300 PP` cutoff

Artifacts:

- [artifacts/final-ownerplan-failure](artifacts/final-ownerplan-failure)
- gallocr reserve divergence evidence: [artifacts/final-ownerplan-galloc-debug/server.log](artifacts/final-ownerplan-galloc-debug/server.log)

This is the decisive local closeout result.

## Code Included

The modified code is included in two forms:

1. Source snapshot of all modified tracked files:
   - [code/source](code/source)
2. Diff metadata:
   - [code/meta/research-changes.patch](code/meta/research-changes.patch)
   - [code/meta/diff-stat.txt](code/meta/diff-stat.txt)
   - [code/meta/modified-files.txt](code/meta/modified-files.txt)
   - [code/meta/base.txt](code/meta/base.txt)

The source snapshot is relative to the working fork:

- base repo: `https://github.com/skyne98/llama.cpp-gfx906.git`
- local HEAD at packaging time is recorded in [code/meta/base.txt](code/meta/base.txt)

## Scripts Included

Research scripts are copied under [scripts](scripts), including:

- `bench-minimax-cache-session.sh`
- `bench-minimax-fattn-tmp-cap.sh`
- `bench-minimax-mmq-diagnostics.sh`
- `bench-minimax-mypack.sh`
- `bench-minimax-placement.sh`
- `bench-minimax-tensor-smoke.sh`
- `bench-qwen-coder-next-cuda.sh`
- `scripts/tools/analyze-moe-profile.py`
- `scripts/tools/build-minimax-fast.sh`

## Recommended Reading Order

1. [docs/minimax-optimization-report.md](docs/minimax-optimization-report.md)
2. [docs/minimax-external-ai-prompt.md](docs/minimax-external-ai-prompt.md)
3. [commands/best-cold-pp.sh](commands/best-cold-pp.sh)
4. [commands/best-cache-session.sh](commands/best-cache-session.sh)
5. [artifacts/final-ownerplan-failure](artifacts/final-ownerplan-failure)
6. [code/meta/research-changes.patch](code/meta/research-changes.patch)

## Practical Conclusion

The local plan is exhausted.

If work resumes, the next branch should be:

- deeper scheduler/backend architecture

It should not be:

- another bounded Stage 5A proof
- another owner-remap taste
- another graph-level packed-prefix proof
- another one-group GQA proof
