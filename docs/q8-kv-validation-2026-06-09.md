# Q8 KV Validation 2026-06-09

Follow-up validation after the main mixed CUDA+ROCm research handoff.

Goal:

- keep full context at `69376`
- switch KV to `q8_0`
- stay on the strongest known mixed command family as much as practical
- verify a real cold run and a real cache-reuse smoke
- leave one tested server command for OpenCode

## Result

This works on the current custom fork without needing a special new code build.

The current working Q8 KV server shape is:

- `-c 69376`
- `-ctk q8_0 -ctv q8_0`
- `-sm layer`
- `-dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2`
- `--placement-policy balanced`
- `--device-speed 0.55,0.55,1.0,1.0,1.0,1.0`
- generated split: `13,13,7,11,11,7`
- `--output-device CUDA3`
- `-b 200 -ub 200`
- `--reserve-pp-outputs 1`
- `--cuda-fattn-mma-q4-0-direct --cuda-fattn-mma-q4-0-direct-mode all`
- `--cache-prompt --cache-ram 32768`

## Startup envelope

Using the strongest q4-style command family with Q8 KV:

- `b256` and above failed
- `b228` was startup-stable but request-time OOMed on the full 11521-token cold payload
- `b200` was the highest request-stable shape that was validated end-to-end

## Cold validation

Artifact:

- [artifacts/q8-cold-validation](../artifacts/q8-cold-validation)

Result:

- prompt tokens: `11521`
- predicted tokens: `128`
- prompt throughput: `309.394175 PP`
- generation throughput: `30.657956 TG`

This was run with the exact benchmark-style FA cap:

- `GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=321`
- `GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=321`
- `GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=1`

Reference launch script:

- [scripts/bench-minimax-q8-best-cold.sh](../scripts/bench-minimax-q8-best-cold.sh)

## Cache smoke validation

Artifact:

- [artifacts/q8-cache-smoke](../artifacts/q8-cache-smoke)

This was a smaller two-turn cache-aware smoke on the same Q8 server family, using the portable FA cap for general prompts.

Summary:

- turn 1:
  - `context_tokens=3838`
  - `prompt_n=3742`
  - `processed_pp=239.396730`
- turn 2:
  - `cache_n=3837`
  - `prompt_n=898`
  - `processed_pp=216.089474`

Aggregate cached-turn summary:

- `cache_ratio_cached_turns = 0.810348`
- `processed_pp_cached_turns = 216.089474`
- `effective_context_pp_cached_turns = 1139.402736`
- `tg_cached_turns = 40.031108`

Interpretation:

- prompt cache reuse is working
- Q8 KV is usable for OpenCode-style repeated turns
- this is a functionality/stability result, not a new cache-throughput winner

## Q8 cache route that failed

Adding the q4-era all-CUDA MiniMax FA routing back on top of the tested Q8 server shape did not survive startup on the tested path.

That means the stable OpenCode server recommendation remains:

- no `--minimax-fattn-devices` override
- use the plain balanced Q8 server shape

## Recommended OpenCode server

Reference launch script:

- [scripts/run-minimax-q8-opencode.sh](../scripts/run-minimax-q8-opencode.sh)

This uses the portable FA cap:

- `GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=2`
- `GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=799`
- `GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=1`

This is the safer choice for mixed prompt chunk shapes than the exact `Q=321` benchmark cap.
