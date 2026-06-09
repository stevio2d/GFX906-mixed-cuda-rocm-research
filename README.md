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

Artifacts:

- [artifacts/best-cold-pp](artifacts/best-cold-pp)
- [artifacts/best-cold-pp-repeat](artifacts/best-cold-pp-repeat)

Runnable reference:

- [commands/best-cold-pp.sh](commands/best-cold-pp.sh)
- required ROCm env helper: [commands/env-rocm-gfx906.sh](commands/env-rocm-gfx906.sh)

### Highest real cache-session PP

Best real cache-session cached follow-up result found in the main benchmark family:

- cached turn `processed_pp = 255.317006`
- cached turn `effective_context_pp = 1036.021386`
- cached turn `TG = 3.405396`

Important:

- this is the highest cached follow-up PP that was recorded in the main cache-session runs
- it still failed the project promotion gate of `>300 PP`

Artifacts:

- [artifacts/best-cache-session](artifacts/best-cache-session)
- comparison baseline: [artifacts/cache-baseline](artifacts/cache-baseline)

Runnable reference:

- [commands/best-cache-session.sh](commands/best-cache-session.sh)
- required ROCm env helper: [commands/env-rocm-gfx906.sh](commands/env-rocm-gfx906.sh)

### Q8 KV validated server

Follow-up Q8 KV validation was completed after the main handoff.

Key result:

- full context `69376`
- `q8_0` KV
- tested cold run at `309.394175 PP / 30.657956 TG`
- tested two-turn cache smoke with real reuse

Reference docs and scripts:

- [docs/q8-kv-validation-2026-06-09.md](docs/q8-kv-validation-2026-06-09.md)
- [scripts/run-minimax-q8-opencode.sh](scripts/run-minimax-q8-opencode.sh)
- [scripts/bench-minimax-q8-best-cold.sh](scripts/bench-minimax-q8-best-cold.sh)

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
