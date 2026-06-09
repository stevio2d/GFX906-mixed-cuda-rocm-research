Out‑of‑the‑box optimization leads for MiniMax‑M2.7 on mixed CUDA + ROCm
Background and goal

Your current MiniMax‑M2.7 set‑up already implements manual layer placement, balanced device speeds, KV quantization, large batch sizes and flash‑attention.
Even with these optimizations you see ~459 tokens/s prompt processing (PP) and ~31 tokens/s token generation (TG) at a context of 69376 tokens.
The MI50 cards offer high HBM2 bandwidth but low compute, while the RTX 3090/5070 Ti cards deliver high compute but limited memory.
Future improvements will therefore need to go beyond flag tuning and modify llama.cpp itself.

Recent research on large‑scale mixture‑of‑experts (MoE) serving highlights new directions:
— Disaggregated expert parallelism replicates attention modules on memory‑rich GPUs while placing the feed‑forward (expert) modules on compute‑rich GPUs.
— Ping‑pong micro‑batch scheduling keeps attention and FFN devices busy and hides communication latency.
— Dynamic gating and expert buffering reduce the number of active experts and keep only “hot” experts in GPU memory.
— CPU‑assisted pipeline parallelism offloads sampling and metadata preparation to underutilized CPUs, increasing GPU utilization.
— Optimized communication scheduling orders token transfers to avoid bandwidth contention and assigns experts to GPUs proportional to their compute capacity, producing up to 3.54× speed‑ups on heterogeneous clusters.
These ideas suggest several code‑level leads for llama.cpp that could help you approach the 1000 tokens/s goal without reducing the Q4 quantization or the 69 k token context.

Each proposal below includes its rationale, likely code locations in llama.cpp, expected upside, risks, and a small experiment to validate it.

1 Disaggregate attention and experts across devices

Why it may help:
MegaScale‑Infer shows that replicating the attention module on memory‑rich GPUs and assigning MoE FFN (expert) modules to compute‑rich GPUs improves GPU utilization and allows heterogeneous deployment. For your set‑up, the MI50 cards can host replicated attention layers and the 3090/5070 Ti cards can host the experts. This ensures the key–value cache and attention activations reside on high‑bandwidth memory, while the heavy expert computations run on fast CUDA hardware.

Code areas to inspect/modify:

llama.cpp and llama.h: add a sub‑layer placement mode where each layer is split into attention and FFN modules.
backend/llama-cuda.cu and backend/llama-rocm.cpp: implement device‑aware dispatch functions for attention and FFN.
sampling.cpp: update the scheduler to dispatch attention compute to MI50 streams and expert compute to CUDA streams.

Expected upside:
By moving the memory‑bound attention to MI50 and the compute‑bound FFN to CUDA, you may reduce stalls caused by slow MI50 compute. MegaScale‑Infer achieved up to 1.9× throughput improvement; a conservative expectation for your smaller 7B model is 10–20 % increase in PP, plus a higher ubatch ceiling.

Risks/complexity:

Requires significant refactoring of the core forward pass.
Cross‑device data transfers may bottleneck on PCIe; you will need to use asynchronous cudaMemcpyPeerAsync/hipMemcpyPeerAsync or pinned host buffers.
Maintaining deterministic ordering of MoE tokens across devices can be tricky.

Minimal experiment:
Implement a prototype where a few early layers use attention on ROCm0 and FFN on CUDA0. Benchmark prefill throughput and compare with baseline.

Acceptance criterion:
At least +10 % PP increase or equivalent wall‑clock reduction at the same context size and batch, without degrading TG throughput.

2 Ping‑pong micro‑batch scheduling

Why it may help:
Disaggregated attention introduces idle periods when either the attention or the FFN waits for the other. Splitting the request into micro‑batches and alternating (“ping‑pong”) between attention and FFN hides communication latency and keeps all devices busy.

Code areas to inspect/modify:

scheduler.cpp: add a micro‑batch scheduler that splits a batch of ubatch tokens into smaller sub‑batches (e.g. 8 × 52 tokens).
llama.cpp: update the inference loop so that attention and FFN operations on different micro‑batches run concurrently in separate streams. Use CUDA/ROCm events to synchronize.

Expected upside:
Micro‑batching can improve GPU utilization and overlap compute with transfers. MegaScale‑Infer reports ~1.7× throughput per cost; on your hardware, a 5–15 % PP increase is plausible.

Risks/complexity:

Additional latency for the first tokens because micro‑batches wait for each other.
The existing -ub parameter may need to be repurposed or supplemented.

Minimal experiment:
Implement a ping‑pong scheduler for two sub‑batches and test with -ub 416 on the current best layer split. Measure whether PP increases without harming TG.

Acceptance criterion:
PP increases by ≥5 % relative to the baseline, and micro‑batch overhead doesn’t reduce TG throughput.

3 Implement dynamic gating and load‑aware expert assignment

Why it may help:
The gating network in MoE models often assigns tokens to a fixed number of experts (e.g. top‑2), which can overload popular experts. The dynamic gating proposed by Huang et al. reduces communication and memory cost by matching each expert’s computational capacity to its token assignments. It improved maximum throughput by 6.21–11.55× in language‑model benchmarks. In your scenario, dynamic gating could route more tokens to the faster CUDA experts while limiting assignments to the slower MI50 experts.

Code areas to inspect/modify:

moe_router.cpp or equivalent gating layer in your fork. Modify the top‑k selection to a load‑aware strategy that computes a per‑expert capacity (based on recent runtime statistics) and assigns tokens proportionally.
Add instrumentation to track expert invocation counts and device occupancy.

Expected upside:
If most tokens are routed to fast experts and MI50 experts are engaged only for overflow, communication overhead and MI50 compute time drop. Throughput gains could be 20–50 % when load imbalance is currently severe. Even modest improvements may allow larger ubatch sizes.

Risks/complexity:

Altering the gating function may slightly change output; you should verify that quality remains acceptable.
Requires additional runtime statistics and may introduce per‑token overhead.

Minimal experiment:
Instrument expert activation counts during a typical prompt. Implement a simple heuristic that avoids routing to the two slowest experts when other experts are under‑utilized. Compare PP and TG with and without the heuristic.

Acceptance criterion:
PP increases without measurable degradation in output quality; gating overhead remains below 1 ms per iteration.

4 Expert buffering and CPU offload of cold experts

Why it may help:
Huang et al. propose caching only hot experts in GPU memory and buffering the rest in CPU memory. Because MoE inference activates only a small subset of experts per token, many experts remain unused for long periods. Offloading them to host memory reduces static GPU memory use, allowing larger ubatch sizes and possibly enabling additional CUDA layers.

Code areas to inspect/modify:

Extend the model loader (model-load.cpp) to keep inactive expert matrices on host RAM and upload them to the GPU on demand via asynchronous transfers (cudaMemcpyAsync/hipMemcpyAsync).
Add an LRU cache for expert weights per GPU.

Expected upside:
Freeing even 2–3 GB on each CUDA card may raise the usable ubatch by 1–2 steps, translating to ~5–10 % PP improvement. Reducing memory pressure also lowers the risk of CUDA OOM errors.

Risks/complexity:

On‑demand transfers can add latency when a cold expert becomes active; careful prefetching or small staging buffers are needed.
CPU ↔ GPU bandwidth (PCIe) is limited; repeated streaming of expert weights can negate gains.

Minimal experiment:
Identify the least‑used experts during a typical request and move their weights to pinned host memory. Run a prompt with this configuration and observe memory usage and PP. Compare with baseline where all experts are resident.

Acceptance criterion:
PP or max ubatch increases without a significant (>5 %) drop in TG throughput.

5 High‑performance M2N communication and peer‑to‑peer transfers

Why it may help:
MegaScale‑Infer uses a custom M2N communication library that reduces GPU‑to‑CPU copies and synchronization overhead. In your set‑up, cross‑device transfers (ROCm ↔ CUDA) are currently performed via the CPU. Enabling GPU peer‑to‑peer (P2P) where possible could reduce latency and free CPU bandwidth.

Code areas to inspect/modify:

Investigate whether ROCm–CUDA P2P is supported via PCIe or xGMI. If so, modify ggml-cuda.cu and ggml-rocm.cpp to use cudaMemcpyPeerAsync or hipMemcpyPeerAsync for device‑to‑device copies.
Introduce a small transfer scheduler that batches token dispatches to reduce the number of copy calls.

Expected upside:
Reducing CPU involvement in token routing can shave a few milliseconds off each iteration and improve pipeline overlap. Gains of 5–10 % in PP are possible, especially at high ubatch values.

Risks/complexity:

Current hardware and drivers may not support direct P2P between AMD and NVIDIA GPUs; fallback to pinned CPU buffers may be necessary.
Implementing asynchronous transfers and synchronizing across two driver APIs is error‑prone.

Minimal experiment:
Write a small benchmark that transfers a 16 MB tensor between a ROCm and a CUDA device using both the existing CPU‑mediated path and a potential P2P path. Measure throughput and latency.

Acceptance criterion:
A measurable reduction in transfer latency and at least a 5 % improvement in PP when integrated into inference.

6 Per‑stream memory pools and async memory allocation

Why it may help:
Memory fragmentation currently limits ubatch to 416. vAttention shows that using contiguous virtual memory and on‑demand physical allocation can reduce fragmentation and improve throughput. Using per‑stream memory pools (cudaMallocAsync/hipMallocAsync) can reduce temporary allocations and reuse memory across micro‑batches.

Code areas to inspect/modify:

Replace cudaMalloc/hipMalloc in ggml-cuda.cu and ggml-rocm.cpp with cudaMallocAsync/hipMallocAsync tied to the current stream.
Implement a simple memory pool that returns freed buffers to a cache keyed by size and device.
Ensure proper synchronization when the same buffer is reused across streams.

Expected upside:
By lowering internal fragmentation you can likely raise ubatch beyond 416, directly increasing PP. A 10–20 % PP improvement is plausible if the current limit is memory‑allocator bound.

Risks/complexity:

Asynchronous allocators require recent CUDA/ROCm versions; older drivers may not support them.
Incorrect synchronization can cause subtle bugs or undefined behavior.

Minimal experiment:
Implement a per‑stream allocator in isolation and benchmark repeated allocations and frees for buffers of different sizes. Then integrate into a single layer of llama.cpp and attempt to run with -ub 448 or higher.

Acceptance criterion:
Successful inference at -ub ≥ 448 without OOM, and/or at least 10 % PP gain.

7 Chunked flash‑attention and KV prefetching

Why it may help:
Flash‑attention reduces memory usage by computing attention in blocks, but the current kernel still allocates workspace buffers proportional to the full sequence length. Implementing chunked attention (processing the 69 k context in smaller windows) and prefetching KV cache to CUDA before use can reduce peak VRAM use and allow larger ubatch sizes.

Code areas to inspect/modify:

Modify flash-attn.cu to operate on a sliding window over the KV cache, releasing intermediate buffers after each window.
In llama.cpp, prefetch KV chunks to CUDA using asynchronous copies before the attention call.

Expected upside:
By reducing peak memory usage per attention call, you may be able to increase ubatch or offload additional layers to CUDA. Gains of 5–10 % PP are reasonable.

Risks/complexity:

Chunking adds overhead due to additional kernel launches and copies.
Must ensure numerical parity with unchunked attention.

Minimal experiment:
Implement a proof‑of‑concept where the context is split into two halves. Compare memory consumption and PP with the baseline flash‑attention implementation.

Acceptance criterion:
Lower GPU memory usage allows a higher ubatch without reducing TG throughput.

8 Heuristic scheduler based on max‑flow/graph algorithms

Why it may help:
Aurora’s analysis shows that assigning experts to GPUs in descending order of capacity and ordering token transfers to avoid bandwidth contention can reduce MoE inference time by up to 3.54× on heterogeneous clusters. A scheduler that models devices as nodes in a flow graph and uses a heuristic (e.g. max‑flow or greedy matching) to map layers and experts to devices could balance compute and communication better than the current greedy layer split.

Code areas to inspect/modify:

Add a new --placement-policy graph in llama.cpp that builds a weighted graph where nodes represent layers/expert groups and edges represent possible placements with costs based on compute speed and memory.
Use a simple max‑flow or greedy matching algorithm to assign layers and experts to devices, optionally considering heterogenous NVLink/PCIe bandwidth.

Expected upside:
A better global assignment might free the model to use faster GPUs more effectively and reduce cross‑bus transfers. While the theoretical Aurora algorithm covers extreme cluster sizes, a scaled‑down heuristic could yield 10–15 % PP improvement.

Risks/complexity:

Implementation complexity and overhead of solving the assignment problem at start‑up.
Harder to reason about manual overrides; may require fine‑tuning per deployment.

Minimal experiment:
Implement a greedy matching where each attention/FFN pair is assigned to the GPU with the best compute‑to‑memory ratio that still has capacity. Compare PP with the existing balanced policy.

Acceptance criterion:
Any measurable PP improvement without increasing TG latency.

9 CPU‑assisted gating and token sampling

Why it may help:
SiPipe shows that pipeline parallelism often leaves CPUs underutilized; offloading auxiliary tasks such as sampling and metadata preparation to CPUs improves overall throughput. In MoE inference, gating and top‑k expert selection involve matrix multiplications that are small compared with the FFN compute. Running these on CPUs (with vectorized AVX2/AVX512) could free GPU compute cycles for attention and FFN.

Code areas to inspect/modify:

Move the gating projection and top‑k selection to a CPU function in moe_router.cpp using std::vector/Eigen or int8 intrinsics.
Use pinned memory to transfer the resulting expert IDs back to the GPU.

Expected upside:
Removing the gating kernel from the GPU reduces small kernel launches and stream synchronization, which can matter at large context sizes. Gains are likely modest (3–8 %), but combined with other optimizations may help reach the throughput target.

Risks/complexity:

Extra CPU workload may increase contention if CPU threads are already heavily used.
Transfer of gating results back to GPU must be carefully synchronized to avoid stalls.

Minimal experiment:
Profile the current gating kernel’s execution time on the GPUs. Implement a CPU version and benchmark end‑to‑end PP and TG. Compare the total compute time with the baseline.

Acceptance criterion:
Any PP improvement without a decrease in TG throughput or increase in CPU contention.

Summary

To progress toward a 1000 tokens/s prompt‑processing throughput on your mixed CUDA + ROCm setup, simple flag tuning is insufficient. The research literature suggests that disaggregating attention and expert modules, using ping‑pong micro‑batch scheduling, dynamic gating with load‑aware expert assignment, expert buffering, high‑performance cross‑device communication, memory pooling, chunked flash‑attention, graph‑based scheduling, and CPU‑assisted gating can each offer incremental improvements. Many of these approaches have been shown to deliver large gains in other systems, and combining several of them could significantly boost throughput on your hardware. Each lead above includes a concrete experiment and acceptance criterion to help you prioritize development and avoid wasted effort.
