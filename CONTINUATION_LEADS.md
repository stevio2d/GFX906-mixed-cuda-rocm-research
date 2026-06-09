# Continuation Leads

This file is the short root-level handoff for anyone continuing the MiniMax mixed CUDA+ROCm work.

## Current status

- Local bounded tuning is finished for this cycle.
- No local Stage 5 branch crossed the real cache-aware `>300 PP` gate.
- Best cold result remains much stronger than best cache-aware result.
- If work continues, it should be deeper scheduler/backend/KV architecture work, not another small routing or graph-level proof.

## Best validated commands

### Best cold PP command

Use:

- [commands/best-cold-pp.sh](/home/stefan/GFX906-mixed-cuda-rocm-research/commands/best-cold-pp.sh)

Best result:

- `558.7603 PP / 31.2581 TG`

Artifacts:

- [artifacts/best-cold-pp](/home/stefan/GFX906-mixed-cuda-rocm-research/artifacts/best-cold-pp)
- [artifacts/best-cold-pp-repeat](/home/stefan/GFX906-mixed-cuda-rocm-research/artifacts/best-cold-pp-repeat)

### Best validated Q8 KV cache OpenCode/server command

Use:

- [scripts/run-minimax-q8-opencode.sh](/home/stefan/GFX906-mixed-cuda-rocm-research/scripts/run-minimax-q8-opencode.sh)

Key settings:

- `-c 69376`
- `-ctk q8_0 -ctv q8_0`
- `-b 200 -ub 200`

Validated cold result:

- `309.394175 PP / 30.657956 TG`

Artifacts:

- [artifacts/q8-cold-validation](/home/stefan/GFX906-mixed-cuda-rocm-research/artifacts/q8-cold-validation)
- [artifacts/q8-cache-smoke](/home/stefan/GFX906-mixed-cuda-rocm-research/artifacts/q8-cache-smoke)

## Best cache-aware result that was actually validated

From the curated published artifacts:

- cached follow-up `processed_pp = 255.317006`
- `effective_context_pp = 1036.021386`
- `TG = 3.405396`

This is still below the project gate of `>300 PP`.

Use:

- [commands/best-cache-session.sh](/home/stefan/GFX906-mixed-cuda-rocm-research/commands/best-cache-session.sh)

Artifacts:

- [artifacts/best-cache-session](/home/stefan/GFX906-mixed-cuda-rocm-research/artifacts/best-cache-session)

## Highest-value future leads

Only these are worth reopening.

### 1. Scheduler / gallocr pipeline-parallel reserve blow-up

Why it matters:

- Shared prompt-owner experiments exposed a real allocator divergence:
  - reserve-size measurement was about `648.81 MiB` on `CUDA0`
  - first real pipeline-parallel alloc jumped to about `9043.90 MiB`
  - no-pipeline retry dropped to about `939 MiB` and loaded

Why this is still live:

- This is a concrete backend/scheduler failure mode, not generic brainstorming.
- It is the clearest remaining structural blocker discovered locally.

Where to start:

- [docs/minimax-optimization-report.md](/home/stefan/GFX906-mixed-cuda-rocm-research/docs/minimax-optimization-report.md)
  - see the final Stage `5B.0` entries
- [docs/minimax-external-ai-prompt.md](/home/stefan/GFX906-mixed-cuda-rocm-research/docs/minimax-external-ai-prompt.md)
  - see the Stage `5A Refresh` / wrap sections

### 2. Deeper KV/backend consumer architecture

Why it matters:

- Graph-level packed-prefix proofs established that the layout issue is real.
- But repeated graph-time repack or graph-time packed-shadow reads regressed PP badly.
- If this continues at all, it has to move below the graph-proof layer.

What that means:

- persistent packed-prefix reuse
- backend-native consumer path
- no repeated graph-time `ggml_cont()` repacks

### 3. Deeper executor/ownership architecture

Why it matters:

- Narrow routing, owner remaps, KV-local, token split, and one-group GQA proofs are closed.
- The remaining idea space is broader phase-aware ownership/executor design, not another local tweak.

Constraint:

- Any continuation has to preserve cache-aware behavior, avoid CPU fallback, and avoid long-span `F32` KV materialization.

## Paths that should stay closed

Do not spend more local time on:

- callback-level reroutes
- narrow owner-remap churn
- token-domain exact split as a PP path
- one-group GQA proof as a PP path
- graph-level repeated `q4_0` repack variants
- graph-level packed-prefix read/write proofs
- more MoE-only tuning
- more layer-split retuning

## Best starting docs

Read these first:

- [README.md](/home/stefan/GFX906-mixed-cuda-rocm-research/README.md)
- [RELEASE_NOTES.md](/home/stefan/GFX906-mixed-cuda-rocm-research/RELEASE_NOTES.md)
- [docs/minimax-optimization-report.md](/home/stefan/GFX906-mixed-cuda-rocm-research/docs/minimax-optimization-report.md)
- [docs/minimax-external-ai-prompt.md](/home/stefan/GFX906-mixed-cuda-rocm-research/docs/minimax-external-ai-prompt.md)
- [docs/q8-kv-validation-2026-06-09.md](/home/stefan/GFX906-mixed-cuda-rocm-research/docs/q8-kv-validation-2026-06-09.md)

## Bottom line

- Best cold command is documented and reproducible.
- Best validated Q8 KV server command is documented and reproducible.
- No validated cache-aware winner crossed `300 PP`.
- Future work is deeper architecture only.
