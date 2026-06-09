Yes. For what you want, I would **not** make a MiniMax-specific llama.cpp fork. I would make a **generic MoE execution planner** inside llama.cpp/ggml that treats routed experts as a separate resource class from dense/attention tensors.

The key goal should be:

**Always-active tensors stay on GPU. Routed experts become a managed VRAM/RAM resource with dynamic placement per expert, per layer, per microbatch.**

That is the difference between a normal CPU/GPU offload fork and one that can actually improve **PP** on any MoE.

## Current baseline to build from

Mainline llama.cpp already has the right foundation: `--cpu-moe`, `--n-cpu-moe`, `--override-tensor`, `--op-offload`, multi-GPU split modes, batch/ubatch controls, and KV quant/offload controls. The CLI docs list `--cpu-moe` as “keep all MoE weights in CPU” and `--n-cpu-moe` as keeping the first N MoE layers in CPU; they also expose layer/tensor split modes and `--op-offload`. ([GitHub][1])

The current multi-GPU model is still fairly coarse. llama.cpp documents `layer` split as the default pipeline-parallel mode and `tensor` split as experimental; it also notes that pipeline parallelism is usually better for batch throughput/prefill, while tensor parallelism is more latency-oriented and interconnect-sensitive. ([GitHub][2])

For MoE specifically, a current llama.cpp issue describes the existing hook you want: CPU MoE tensors can be selectively copied CPU→GPU by reading the runtime expert IDs in `ggml_backend_sched_compute_splits()`, but the GPU-side copy currently mirrors the full expert tensor layout and does not persist a compact hot-expert cache across passes. ([GitHub][3])

So your fork should not start by rewriting the model loaders. Start at the **routed expert matmul path**.

---

# The big design: generic MoE planner

I would add a new internal subsystem:

```text
ggml-moe-planner
ggml-moe-cache
ggml-moe-dispatch
ggml-moe-profile
```

Conceptually:

```text
Dense / attention / shared expert tensors:
    GPU-resident whenever possible

Routed expert tensors:
    Some full experts resident in VRAM
    Some experts staged from RAM to GPU for PP
    Some experts computed on CPU
    Some experts cached in VRAM for TG
```

Do **not** hard-code MiniMax, Qwen, DeepSeek, Mixtral, GPT-OSS, Kimi, etc. The generic abstraction should be:

```cpp
struct ggml_moe_expert_group {
    int layer;
    int n_expert;
    int n_active;          // top-k if known from metadata, otherwise runtime
    int hidden_size;
    int ffn_size;
    ggml_type quant_type;

    // tensors that form one routed expert FFN
    ggml_tensor * w_gate;
    ggml_tensor * w_up;
    ggml_tensor * w_down;

    // optional, depending on architecture
    ggml_tensor * w_gate_inp;
    ggml_tensor * w_bias;
};
```

But the discovery should be graph-based, not name-based:

```text
Find GGML_OP_MUL_MAT_ID nodes
Group their expert tensors by layer
Infer expert axis / expert stride
Attach tensor role if possible
Fallback to tensor-name regex only when graph metadata is not enough
```

That makes it apply to any routed-expert model.

---

# The most important PP insight

For **token generation**, MoE sparsity helps a lot because one token uses only top-k experts per layer.

For **prompt processing**, a big microbatch often touches **many or all experts** in each layer. So the naive idea:

```text
“Only copy active experts to GPU”
```

can collapse into:

```text
“Copy nearly all experts to GPU every ubatch”
```

That is why ik_llama.cpp or mainline op-offload may not always show gains. A community llama.cpp MoE guide notes that prompt processing is extremely sensitive to batch/ubatch size because CPU-assigned weights may be copied to GPU for prompt processing; it also notes that llama.cpp’s default offload threshold has historically been around 32 tokens and may be too low when hundreds of GB of CPU weights must cross PCIe. ([Hugging Face][4])

So for highest PP, your planner needs to decide:

```text
For this layer and this ubatch:
    Which experts are used?
    How many tokens go to each expert?
    Is it cheaper to:
        A) run that expert on CPU,
        B) copy that expert to GPU and run it,
        C) use an already-resident GPU expert,
        D) send token activations to another GPU that owns the expert?
```

That is the core of the fork.

---

# Priority 1: add MoE profiling first

Before optimization, add measurement. You want the fork to print this per prompt:

```text
MoE profile:
  layer 00: unique experts 37/128, top expert tokens 412, H2D 1.8 GB
  layer 01: unique experts 42/128, top expert tokens 390, H2D 2.0 GB
  ...
  total expert H2D: 117 GB
  total CPU expert time: 4.2 s
  total GPU expert time: 1.3 s
  expert cache hits: 64%
  op-offload triggered: yes
  pp tokens/s: 143
```

Add counters for:

```cpp
struct ggml_moe_stats {
    uint64_t n_tokens;
    uint64_t n_layers;
    uint64_t n_expert_selections;
    uint64_t n_unique_experts;
    uint64_t h2d_bytes_expert;
    uint64_t d2h_bytes;
    uint64_t cpu_expert_us;
    uint64_t gpu_expert_us;
    uint64_t h2d_expert_us;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
};
```

Expose:

```bash
--moe-profile
--moe-profile-json moe-profile.json
--moe-dump-layer-stats
```

Without this, you are guessing.

---

# Priority 2: adaptive PP op-offload threshold

Mainline already has `--op-offload`, and community tuning guides discuss `GGML_OP_OFFLOAD_MIN_BATCH` as the key knob for hybrid MoE PP. ([Hugging Face][4])

Make this first-class and automatic:

```bash
--moe-pp-offload auto
--moe-pp-min-batch auto
--moe-pp-min-batch 512
--moe-pp-max-copy-gb 24
```

The decision should not be:

```cpp
if (n_tokens >= 32) copy CPU ops to GPU;
```

It should be:

```cpp
T_cpu  = estimate_cpu_expert_time(layer, experts_used, tokens_per_expert);
T_copy = bytes_to_stage / measured_h2d_bw;
T_gpu  = estimate_gpu_expert_time(layer, experts_used, tokens_per_expert);

if (T_copy + T_gpu < T_cpu) {
    stage_to_gpu_and_run();
} else {
    run_cpu_gpu_hybrid();
}
```

A simple first version:

```cpp
bool should_stage_expert_pp(
    size_t expert_bytes,
    int tokens_for_expert,
    double h2d_gb_s,
    double cpu_tok_s,
    double gpu_tok_s
) {
    double t_copy = expert_bytes / (h2d_gb_s * 1e9);
    double t_cpu  = tokens_for_expert / cpu_tok_s;
    double t_gpu  = tokens_for_expert / gpu_tok_s;
    return t_copy + t_gpu < t_cpu;
}
```

Then replace the constants with runtime-calibrated measurements.

This is likely one of the biggest PP wins.

---

# Priority 3: compact expert staging, not full mirrored tensors

The llama.cpp issue I found describes today’s limitation clearly: the GPU-side temporary expert tensor mirrors the full CPU expert layout; selected experts are copied into their global offsets, then discarded. The proposed improvement is a persistent GPU slot buffer where `expert_id → slot_id`, and kernels receive remapped slot IDs. ([GitHub][3])

That is exactly what your fork should implement, but make it generic.

Current-style concept:

```text
GPU buffer shape:
    [all experts]

Copy expert 37 into slot 37
Copy expert 91 into slot 91
Most of buffer unused
Discard after pass
```

Better:

```text
GPU staging/cache buffer:
    [N compact slots]

expert 37 → slot 0
expert 91 → slot 1
expert 12 → slot 2
...
```

Add:

```bash
--moe-gpu-slots 256
--moe-gpu-slots-per-layer auto
--moe-gpu-slot-budget 18G
```

Internal state:

```cpp
struct ggml_moe_slot {
    int device;
    int layer;
    int expert_id;
    int slot_id;
    size_t bytes;
    uint64_t last_used;
    uint32_t hit_count;
    bool resident;
    bool dirty;
};

struct ggml_moe_cache {
    std::vector<ggml_moe_slot> slots;
    std::unordered_map<uint64_t, int> expert_to_slot; // key = layer/expert/tensor-role
};
```

For the GPU kernel, create a remapped ID tensor:

```text
global expert IDs:
    [37, 91, 12, 37, ...]

slot IDs:
    [0, 1, 2, 0, ...]
```

Then `GGML_OP_MUL_MAT_ID` reads from compact slot IDs rather than global expert IDs.

This change benefits both PP and TG, but PP still needs an anti-pollution policy.

---

# Priority 4: separate PP staging cache from TG hot cache

Do **not** use plain LRU for everything.

A large prompt can touch almost every expert and evict the experts that matter for decode. The two-tier expert-cache issue proposes SLRU plus a frequency-gated admission filter because prefill can wipe a normal LRU cache; it also reports a PoC where a protected cache survived prefill better and achieved much higher steady-state hit rate than pure CPU offload. ([GitHub][3])

Use two caches:

```text
PP scratch cache:
    large, temporary, optimized for current ubatch
    can be destroyed after prompt processing

TG hot cache:
    persistent, protected, optimized for repeated decode experts
    should not be polluted by one-off prefill experts
```

CLI:

```bash
--moe-cache on
--moe-cache-policy slru
--moe-cache-protected-ratio 0.80
--moe-cache-admit-after 2
--moe-pp-cache-policy scratch
--moe-pp-preserve-tg-cache on
```

Policy:

```text
During PP:
    Use scratch slots first.
    Do not admit one-time experts into protected TG cache.
    Admit only experts that repeat across layers/ubatches/prompts.

During TG:
    Use SLRU/LFU.
    Promote experts after repeated hits.
```

This is especially important for 128k context workflows where a huge prefill is followed by long generation.

---

# Priority 5: expert-parallel multi-GPU for PP

This is where the biggest long-term gain is.

Current llama.cpp multi-GPU is mostly layer/tensor split. For mixed consumer GPUs, layer split is compatible and good for memory capacity, but it often leaves only one GPU doing useful work at a time for parts of a single request. The docs describe layer split as pipeline parallelism and tensor split as experimental and interconnect-sensitive. ([GitHub][2])

For MoE PP, you want a third strategy:

```text
--split-mode moe-expert
```

or:

```text
--moe-expert-parallel on
```

Instead of splitting layers, split experts:

```text
Layer 0:
    GPU0 owns experts 0-63
    GPU1 owns experts 64-127
    GPU2 owns experts 128-191
    CPU owns cold overflow experts

Layer 1:
    same or route-profile-balanced split

...
```

Then for each MoE layer:

1. Router produces top-k expert IDs.
2. Build token buckets per expert.
3. Send token activations to the device that owns each expert.
4. Each GPU computes its local expert groups in parallel.
5. Gather weighted outputs back.

Why this can beat weight streaming:

```text
Copying expert weights:
    huge, many GB per layer/ubatch

Sending token activations:
    B × top_k × hidden × bytes
    usually much smaller than copying all expert weights
```

For PP, this is usually the right direction. You want to move **activations**, not **weights**.

Add:

```bash
--moe-expert-parallel auto
--moe-expert-shard balanced
--moe-expert-shard route-profile
--moe-expert-shard-vram 24,24,16,16
--moe-expert-cpu-overflow on
```

The hard part is dynamic routing and gather, but it is generic. Every MoE model has the same core pattern:

```text
hidden states → router → expert IDs → expert FFN → weighted sum
```

So you can support many MoEs with one executor.

---

# Priority 6: dynamic ubatch, not just “make ubatch huge”

For dense models, bigger `-ub` often improves PP until VRAM runs out.

For MoE, bigger `-ub` also increases the number of unique experts touched per layer. There is a point where a bigger ubatch makes you copy/dispatch too much.

Add:

```bash
--moe-auto-ubatch on
--moe-ubatch-min 256
--moe-ubatch-max 8192
--moe-ubatch-target unique_experts
```

Heuristic:

```cpp
if (unique_expert_ratio > 0.80 && h2d_bytes_high) {
    reduce_ubatch();
}

if (unique_expert_ratio < 0.40 && GPU underutilized) {
    increase_ubatch();
}
```

Better cost target:

```text
Choose ubatch that minimizes:
    max(cpu_expert_time, gpu_expert_time)
  + h2d_copy_time
  + gpu_attention_time
  + dispatch/gather_time
```

For your hardware, this matters a lot because your RAM bandwidth is high for a server but still far below aggregate GPU compute.

---

# Priority 7: pinned RAM and async transfer, carefully

You want CPU RAM to be a real backing store, not pageable memory that CUDA has to stage internally.

But do not blindly pin 150–400 GB of RAM. That can hurt the OS and can reduce stability.

Use:

```text
Pinned staging pool per GPU
Pinned hot-expert pool
Optional lazy cudaHostRegister for frequently copied expert pages
```

CLI:

```bash
--moe-pinned-staging 4G
--moe-pinned-hot 16G
--moe-host-register-hot on
--moe-host-register-all off
```

Implementation idea:

```cpp
struct ggml_moe_pinned_pool {
    int device;
    void * ptr;
    size_t size;
    std::atomic<size_t> cursor;
    cudaStream_t h2d_stream;
};
```

Use multiple CUDA streams:

```text
stream_compute
stream_h2d
stream_d2d/p2p
stream_gather
```

Overlap:

```text
While GPU computes layer L expert group A:
    Copy layer L expert group B
    CPU computes cold expert group C
    Pre-pack layer L+1 route buckets
```

Without overlap, expert staging will look much worse than it should.

---

# Priority 8: CPU and GPU should run experts concurrently

A lot of CPU/GPU offload engines accidentally serialize:

```text
GPU work
then CPU work
then GPU work
then CPU work
```

For hybrid MoE you want:

```text
GPU-resident experts execute on GPU
Staged hot experts execute on GPU
Cold / low-token experts execute on CPU
Then gather outputs
```

For one layer:

```cpp
moe_plan = build_plan(layer, ids, tokens);

launch_gpu_experts_async(moe_plan.gpu_resident);
launch_h2d_and_gpu_experts_async(moe_plan.gpu_staged);
launch_cpu_experts_async(moe_plan.cpu);

wait_all();
gather_weighted_outputs();
```

This can improve PP because small cold experts may not be worth copying to GPU, but the CPU can process them while GPUs are busy.

---

# A practical cost model

For each expert `e` in layer `l`:

```text
S_e = expert weight bytes
N_e = number of token assignments for this expert in current ubatch
```

Estimate:

```text
GPU resident:
    T = gpu_matmul_time(N_e)

GPU staged:
    T = S_e / H2D_bandwidth + gpu_matmul_time(N_e)

Remote GPU expert-parallel:
    T = activation_bytes(N_e) / link_bandwidth
      + gpu_matmul_time(N_e)
      + output_bytes(N_e) / link_bandwidth

CPU:
    T = cpu_expert_time(N_e)
```

Pick the lowest-cost path, but also consider device queue load:

```cpp
score(device) = predicted_finish_time(device) + transfer_time + compute_time;
```

Then assign each expert to the device with the lowest score.

This lets the same system work on:

```text
1 GPU + lots of RAM
2 GPUs with NVLink
4 mixed GPUs over PCIe
Dual socket CPU
EPYC single socket
Large VRAM workstation
```

---

# Suggested CLI surface

I would expose this as experimental flags:

```bash
--moe-mode auto|cpu|gpu-stage|expert-parallel|hybrid
--moe-profile
--moe-profile-json FILE

--moe-pp-policy auto|cpu-hybrid|gpu-stage|expert-parallel
--moe-pp-min-batch auto|N
--moe-auto-ubatch on|off
--moe-ubatch-min N
--moe-ubatch-max N

--moe-cache on|off
--moe-cache-policy lru|slru|lfu|fifo
--moe-cache-slots N
--moe-cache-vram MiB
--moe-cache-admit-after N
--moe-cache-protected-ratio F

--moe-pinned-staging MiB
--moe-host-register-hot on|off

--moe-expert-parallel on|off|auto
--moe-expert-shard balanced|route-profile|manual
--moe-expert-cpu-overflow on|off
```

Example for your goal:

```bash
./llama-server \
  -m model.gguf \
  -ngl all \
  --cpu-moe \
  -fa on \
  -b 8192 \
  -ub 4096 \
  --moe-mode hybrid \
  --moe-pp-policy auto \
  --moe-auto-ubatch on \
  --moe-cache on \
  --moe-cache-policy slru \
  --moe-cache-vram 12000 \
  --moe-pinned-staging 4096 \
  --moe-expert-parallel auto \
  --moe-profile
```

---

# What I would not copy blindly from ik_llama.cpp

ik_llama.cpp has useful ideas: hybrid CPU/GPU performance work, fused MoE operations, tensor overrides, row-interleaved quant packing, graph split, and extra quant types. Its README explicitly says it targets better CPU and hybrid GPU/CPU performance, including MoE and fused MoE operations. ([GitHub][5])

But one warning is important for your case: ik’s README says row-interleaved repacking can hurt hybrid MoE prompt processing because not all quant types have CUDA row-interleaved implementations, and it specifically mentions k-quants like Q3_K/Q4_K/Q5_K/Q6_K. ([GitHub][5])

So for your fork:

```text
Do not make CPU-only repacking the default.
Do not create a quant layout that loses CUDA kernels.
Do not optimize TG at the cost of PP.
```

Your PP-oriented fork should keep expert weights in layouts that can be:

```text
1. read efficiently by CPU,
2. copied/staged efficiently,
3. consumed by CUDA kernels without repacking every batch.
```

---

# Minimal implementation roadmap

## Phase 0 — reproducible benchmarks

Add a script that compares:

```text
mainline llama.cpp
ik_llama.cpp
your fork
```

Across:

```text
Mixtral 8x7B
Qwen3 MoE / Qwen3-30B-A3B style model
DeepSeek-V2-Lite or similar
GPT-OSS-120B if available
MiniMax only as one test, not the target
```

Benchmark:

```text
PP 512
PP 4096
PP 32768
PP 128k if model supports it
TG at 4k
TG at 64k
TG after large prefill
```

Track:

```text
tok/s
TTFT
H2D GB/s
CPU RAM GB/s
GPU utilization
unique experts/layer
cache hit rate
```

## Measured hardware baseline on llmserver01

Test date: 2026-06-05.

The host is a single-socket AMD EPYC 7532 system with 4 NUMA nodes and 32 physical cores / 64 threads. Memory inventory is clean:

```text
8 populated channels: P0 CHANNEL A-H
8 x 32 GB Micron 36ASF4G72PZ-2G6E1
DDR4 RDIMM, dual-rank, ECC
Reported speed: 2666 MT/s
Configured memory speed: 2666 MT/s
Configured voltage: 1.2 V
EDAC corrected/uncorrected DIMM errors: 0
```

Raw DDR4-2666 pin-rate is:

```text
2666 MT/s * 8 bytes * 8 channels = ~170.6 GB/s
```

That is not the practical inference bandwidth. The valid STREAM-style DRAM results were:

```text
Original OpenMP binding: ~75-80 GB/s Triad
Cleaner physical-core binding: ~105 GB/s Triad
Per-NUMA-quadrant local result: ~26.5 GB/s Triad
Best all-socket result observed: ~105.4 GB/s Triad
```

The cleaner benchmark command shape was:

```bash
OMP_NUM_THREADS=32 \
OMP_PROC_BIND=false \
GOMP_CPU_AFFINITY=0-31 \
taskset -c 0-31 \
/home/stefan/bin/stream_bandwidth
```

For actual llama.cpp CPU-only decode, allocation policy mattered more than CPU governor. TinyLlama 1.1B Q4_K_M measured:

```text
Default CPU-only decode: ~108 tok/s
With non-mmap + NUMA interleave: ~130 tok/s
Effective model-weight streaming at ~130 tok/s: ~86.8 GB/s
```

Recommended CPU/RAM inference command shape:

```bash
CUDA_VISIBLE_DEVICES='' \
numactl --interleave=all \
taskset -c 0-31 \
/home/stefan/builds/llama-mixed/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 0 \
  -t 32 \
  -mmp 0
```

Important interpretation:

```text
Use physical cores only for this workload: 0-31
Avoid SMT for decode unless a larger model proves otherwise.
Use -mmp 0 when CPU/RAM bandwidth matters.
Use numactl --interleave=all to spread anonymous model allocation across the NUMA memory controllers.
Do not expect the CPU path to approach RTX 3090 VRAM bandwidth.
```

RTX 3090 comparison:

```text
RTX 3090 VRAM bandwidth spec: ~936 GB/s
Measured best CPU STREAM Triad: ~105 GB/s, ~11% of RTX 3090 VRAM bandwidth
Measured TinyLlama CPU effective streaming: ~87 GB/s, ~9% of RTX 3090 VRAM bandwidth
TinyLlama on one RTX 3090: ~525 tok/s decode
TinyLlama CPU best: ~130 tok/s decode
```

The installed DIMMs are already running at their reported 2666 MT/s speed. The EPYC 7532 platform can use faster memory than this, but this installed Micron part is a DDR4-2666 RDIMM; moving the RAM ceiling higher requires different DIMMs and BIOS validation, not an OS setting.

## Phase 1 — instrumentation

No behavior changes. Just measure expert usage and copy volume.

Expected result: you will discover which models touch nearly all experts during PP and which keep routing sparse.

## Phase 2 — expose/adapt PP threshold

Make `GGML_OP_OFFLOAD_MIN_BATCH` behavior a real CLI flag, then add auto-calibration.

This is easy and likely gives immediate PP gains on some systems.

## Phase 3 — compact expert staging

Replace full-layout temporary expert GPU tensors with compact slot buffers and ID remapping.

This is the first major architectural win.

## Phase 4 — PP scratch cache + TG protected cache

Implement SLRU or 2Q-style cache.

Make prefill unable to evict all decode-hot experts.

## Phase 5 — CPU/GPU concurrent expert execution

Split expert IDs into:

```text
GPU resident
GPU staged
CPU cold
```

Run them concurrently and gather.

## Phase 6 — expert-parallel multi-GPU

Shard experts across GPUs. Send activations to experts rather than copying expert weights to one GPU.

This is the most work, but it is the biggest potential PP win on multi-GPU systems.

---

# Where to patch

Based on the current architecture and the public issue analysis, the likely hook points are:

```text
ggml/src/ggml-backend.cpp
    ggml_backend_sched_compute_splits()
    current expert-copy path
    GGML_OP_MUL_MAT_ID scheduling

common/arg.cpp
    new CLI flags

src/llama.cpp / src/llama-model*.cpp
    MoE tensor placement
    model-load expert metadata

ggml/src/ggml-cuda/*
    compact expert slot kernels
    remapped MUL_MAT_ID
    async staging streams

tools/llama-bench
    PP/TG MoE profiling
```

The llama.cpp two-tier-cache issue specifically points at `ggml_backend_sched_compute_splits()` and `GGML_OP_MUL_MAT_ID` as the current selective-expert-copy hook, which matches the generic design above. ([GitHub][3])

---

# The first useful patch I would write

The first patch should be boring but valuable:

```bash
--moe-profile
--moe-pp-min-batch
--moe-print-copy-stats
```

Example output:

```text
moe: pp ubatch=4096
moe: layer=12 unique=113/128 selections=32768
moe: h2d experts=5.41 GiB
moe: cpu expert time=212 ms
moe: gpu staged time=154 ms
moe: chosen=gpu-stage
```

Then add:

```bash
--moe-pp-min-batch auto
```

Auto chooses whether op-offload should happen for each ubatch.

This alone may beat ik/mainline in PP on some models because it avoids the bad case:

```text
copy 100+ GB over PCIe for a prompt batch where CPU+GPU hybrid would have finished sooner
```

---

# The real “best use of VRAM + RAM” policy

For your machine and for generic MoEs, the ideal policy is:

```text
VRAM:
    KV cache
    attention
    shared experts
    dense FFN
    router
    hottest routed experts
    PP scratch expert slots

RAM:
    full routed-expert backing store
    cold experts
    overflow expert tensors
    pinned staging pool

GPU interconnect / PCIe:
    move activations when possible
    move weights only when enough tokens amortize the copy
```

That is the main principle.

---

# Expected gains

For PP, I would expect this ordering:

| Change                               |                                  Likely PP gain |
| ------------------------------------ | ----------------------------------------------: |
| Better batch/ubatch defaults only    |                               small to moderate |
| Adaptive op-offload threshold        |                                        moderate |
| Compact staging slots                |                                        moderate |
| Pinned async staging                 |                                        moderate |
| PP scratch cache with anti-pollution | small to moderate for PP, large for TG-after-PP |
| CPU/GPU concurrent expert execution  |                               moderate to large |
| Expert-parallel multi-GPU            |                      large, if implemented well |

The biggest single architectural PP win is probably **expert-parallel multi-GPU**, because it avoids repeatedly moving giant expert weights and instead moves much smaller activation batches.

The biggest near-term/easier win is probably **adaptive PP offload threshold + compact expert staging**.

---

# Bottom line

Build this as a **generic routed-expert runtime**, not a MiniMax runtime.

The fork should make three decisions dynamically for every MoE layer and microbatch:

```text
1. Which experts are used?
2. Where should each expert run: resident GPU, staged GPU, remote GPU, or CPU?
3. Is copying weights cheaper than computing on CPU or sending activations?
```

For highest PP, I would prioritize:

```text
1. MoE profiling
2. adaptive op-offload threshold
3. compact expert staging with ID remap
4. dynamic ubatch
5. CPU/GPU concurrent expert execution
6. expert-parallel multi-GPU
7. protected TG cache so big prefills do not ruin decode speed
```

That design should work across Mixtral-style, Qwen-MoE, DeepSeek-style, GPT-OSS, MiniMax, and future GGUF MoE models, because it targets the common operation: **routed expert matmul**, not the model family.

[1]: https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md "llama.cpp/tools/cli/README.md at master · ggml-org/llama.cpp · GitHub"
[2]: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md "llama.cpp/docs/multi-gpu.md at master · ggml-org/llama.cpp · GitHub"
[3]: https://github.com/ggml-org/llama.cpp/issues/20757 "Feature Request: Two-tier GPU+RAM expert cache for MoE offload (pluggable eviction policy) · Issue #20757 · ggml-org/llama.cpp · GitHub"
[4]: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide "Performant local mixture-of-experts CPU inference with GPU acceleration in llama.cpp"
[5]: https://github.com/ikawrakow/ik_llama.cpp "GitHub - ikawrakow/ik_llama.cpp: llama.cpp fork with additional SOTA quants and improved performance · GitHub"
