# Release Notes

## 2026-06-09 Research Handoff

Published handoff for the MiniMax-M2.7 mixed CUDA+ROCm research cycle.

### Best cold command

Result:

- `558.7603 PP / 31.2581 TG`
- repeat: `558.6422 PP / 31.2530 TG`

Command:

- [commands/best-cold-pp.sh](commands/best-cold-pp.sh)
- required environment: [commands/env-rocm-gfx906.sh](commands/env-rocm-gfx906.sh)

Primary artifacts:

- [artifacts/best-cold-pp](artifacts/best-cold-pp)
- [artifacts/best-cold-pp-repeat](artifacts/best-cold-pp-repeat)

### Best real cache-session command

Best cached follow-up result in the main real cache-session benchmark family:

- cached turn `processed_pp = 255.317006`
- cached turn `effective_context_pp = 1036.021386`
- cached turn `TG = 3.405396`

Important:

- this is the highest cached follow-up PP recorded in the main cache-session runs
- it still fails the real promotion gate of `>300 PP`

Command:

- [commands/best-cache-session.sh](commands/best-cache-session.sh)
- required environment: [commands/env-rocm-gfx906.sh](commands/env-rocm-gfx906.sh)

Primary artifacts:

- [artifacts/best-cache-session](artifacts/best-cache-session)
- baseline comparison: [artifacts/cache-baseline](artifacts/cache-baseline)

### Best controlled exact Stage 5 fixture

Diagnostic only, not the promotion metric:

- turn 2: `408.65 PP`
- turn 3: `386.81 PP`
- turn 4: `365.85 PP`

Artifact:

- [artifacts/best-exact-stage5-fixture](artifacts/best-exact-stage5-fixture)

### Final verdict

The local throughput plan is exhausted for this cycle.

Final bounded branch result:

- shared prompt-owner runtime taste
- branch definitely active in server log
- valid result: turn 1 only `228.09 PP / 17.99 TG`
- stopped below the `300 PP` cutoff before cached follow-up

Artifacts:

- [artifacts/final-ownerplan-failure](artifacts/final-ownerplan-failure)
- allocator divergence evidence: [artifacts/final-ownerplan-galloc-debug/server.log](artifacts/final-ownerplan-galloc-debug/server.log)

Conclusion:

- no bounded local branch crossed the real `>300 PP` cache-aware gate
- any future continuation is deeper scheduler/backend architecture only

### Reference docs

- [docs/minimax-optimization-report.md](docs/minimax-optimization-report.md)
- [docs/minimax-external-ai-prompt.md](docs/minimax-external-ai-prompt.md)
- [docs/plan-cuda+8channelram.md](docs/plan-cuda+8channelram.md)
