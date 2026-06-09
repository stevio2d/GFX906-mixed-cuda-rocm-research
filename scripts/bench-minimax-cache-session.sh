#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/stefan/llama.cpp-gfx906}
BIN=${BIN:-$ROOT/build-cuda-hip-dl-gfx906/bin/llama-server}
MODEL=${MODEL:-/home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf}
DEV=${DEV:-ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2}
CTX=${CTX:-69376}
PORT=${PORT:-21880}
N_PREDICT=${N_PREDICT:-96}
MAX_TURNS=${MAX_TURNS:-10}
TARGET_CONTEXT=${TARGET_CONTEXT:-66000}
BASE_REPEATS=${BASE_REPEATS:-130}
TURN_REPEATS=${TURN_REPEATS:-78}
CACHE_RAM_MIB=${CACHE_RAM_MIB:-32768}
PP_STOP_CUTOFF=${PP_STOP_CUTOFF:-300}
BATCH=${BATCH:-800}
UBATCH=${UBATCH:-$BATCH}
PLACEMENT_POLICY=${PLACEMENT_POLICY:-memory}
DEVICE_SPEED=${DEVICE_SPEED:-0.55,0.55,1.0,1.0,1.0,1.0}
LAYER_SPLIT=${LAYER_SPLIT-14,12,7,11,11,7}
OUTPUT_DEVICE=${OUTPUT_DEVICE-CUDA3}
EXTRA_ARGS=${EXTRA_ARGS:-}
ALLOW_PIPELINE_FALLBACK=${ALLOW_PIPELINE_FALLBACK:-0}
RUN_DIR=${RUN_DIR:-$ROOT/bench-results/minimax-cache-session-$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$RUN_DIR"

export PATH=/opt/rocm/bin:${PATH:-}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.0.6}
export ROCBLAS_LAYER=${ROCBLAS_LAYER:-0}
export ROCBLAS_LOG_LEVEL=${ROCBLAS_LOG_LEVEL:-0}
export HIPBLASLT_LOG_LEVEL=${HIPBLASLT_LOG_LEVEL:-0}
export HIP_FORCE_DEV_KERNARG=${HIP_FORCE_DEV_KERNARG:-1}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-8}
export ROCBLAS_TENSILE_LIBPATH=${ROCBLAS_TENSILE_LIBPATH:-/opt/rocm-custom/lib/rocblas/library}
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Portable form of the current ub800 best: cap non-full HIP prompt chunks, keep full Q=800 chunks fast.
export GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS=${GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS:-hip}
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=${GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN:-2}
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=${GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX:-799}
export GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=${GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS:-1}

TRANSCRIPT="$RUN_DIR/session_transcript.txt"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_JSON="$RUN_DIR/summary.json"
SUMMARY_TXT="$RUN_DIR/summary.txt"
SERVER_LOG="$RUN_DIR/server.log"
TELEMETRY_LOG="$RUN_DIR/telemetry.log"
CMD_FILE="$RUN_DIR/cmd.txt"

printf "turn\tstatus\tcontext_tokens\tcache_n\tprompt_n\tpredicted_n\tcache_ratio\tprocessed_pp\teffective_context_pp\ttg\tprompt_ms\tpredicted_ms\twall_ms\ttokens_cached\tresponse_sha256\tprompt_file\tresponse_file\tslots_file\n" > "$RESULTS_TSV"

python3 - "$TRANSCRIPT" "$BASE_REPEATS" "$TURN_REPEATS" <<'PY'
import sys

path, base_repeats, turn_repeats = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

repo_unit = """\
### Repository snapshot shard
Path: src/session_orchestrator_{i}.ts
```ts
export type PendingEdit{i} = {{
  file: string;
  reason: string;
  oldText: string;
  newText: string;
}};

export function planEdit{i}(input: string): PendingEdit{i} {{
  const normalized = input.trim().replace(/\\s+/g, " ");
  return {{
    file: "src/session_orchestrator_{i}.ts",
    reason: "Keep cache-aware coding benchmark realistic while preserving a large stable prefix.",
    oldText: normalized.slice(0, 96),
    newText: `${{normalized}} :: reviewed in shard {i}`
  }};
}}
```
Notes: this shard appears in the stable project context. It describes code, tests, and constraints that remain mostly unchanged across follow-up turns.

"""

turn_unit = """\
#### Follow-up changed suffix shard {turn}.{i}
The user is iterating on a real coding task. Inspect the patch below, explain the bug risk, and propose the smallest safe edit.

```diff
diff --git a/src/cache_session_{turn}_{i}.ts b/src/cache_session_{turn}_{i}.ts
@@
-export const reuseWindow = {old_window};
+export const reuseWindow = {new_window};
 export function shouldReusePrefix(commonPrefixTokens: number, promptTokens: number): boolean {{
-  return commonPrefixTokens > promptTokens * 0.50;
+  return commonPrefixTokens > promptTokens * 0.82;
 }}
```

Acceptance notes: preserve deterministic output, avoid CPU fallback, keep prompt-cache reuse visible, and record context depth.

"""

with open(path, "w", encoding="utf-8") as f:
    f.write("System: You are an OpenCode coding assistant running inside llama.cpp. ")
    f.write("Answer with concise implementation guidance and patch-level reasoning.\\n\\n")
    f.write("Repository context below is intentionally large and stable to test prompt-cache reuse.\\n\\n")
    for i in range(base_repeats):
        f.write(repo_unit.format(i=i))
    f.write("\\nUser turn 1:\\n")
    f.write("We are optimizing llama.cpp for MiniMax-M2.7 on mixed CUDA and ROCm. Review this current stage and propose the next safe code change.\\n\\n")
    for i in range(turn_repeats):
        f.write(turn_unit.format(turn=1, i=i, old_window=128 + i, new_window=256 + i))
    f.write("\\nAssistant:\\n")
PY

cleanup_server() {
    local pid=${1:-}
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! kill -0 "$pid" 2>/dev/null; then
                return
            fi
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
}

sample_telemetry() {
    local pid=$1
    while kill -0 "$pid" 2>/dev/null; do
        {
            date -u '+=== %Y-%m-%dT%H:%M:%SZ ==='
            ps -o pid,rss,vsz,pcpu,pmem,comm -p "$pid" 2>/dev/null || true
            nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null || true
            rocm-smi --showmeminfo vram --showuse --showbus 2>/dev/null || true
        } >> "$TELEMETRY_LOG"
        sleep 5
    done
}

wait_for_server() {
    local pid=$1
    local fail_re="failed to load|error loading model"
    if [[ "$ALLOW_PIPELINE_FALLBACK" != "1" ]]; then
        fail_re+="|out of memory|unable to allocate|failed to allocate"
    fi
    for _ in $(seq 1 900); do
        if grep -q "server is listening" "$SERVER_LOG" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if grep -qiE "$fail_re" "$SERVER_LOG" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
}

append_next_turn() {
    local turn=$1
    local response_file=$2
    python3 - "$TRANSCRIPT" "$response_file" "$turn" "$TURN_REPEATS" <<'PY'
import json
import sys

transcript, response_file, turn, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

try:
    with open(response_file, "r", encoding="utf-8") as f:
        content = json.load(f).get("content", "")
except Exception:
    content = ""

turn_unit = """\
#### Follow-up changed suffix shard {turn}.{i}
The coding session continues. The stable repository prefix is unchanged; only this suffix changes as the user adds a new task.

```diff
diff --git a/src/cache_session_{turn}_{i}.cpp b/src/cache_session_{turn}_{i}.cpp
@@
-int copy_batch = {old_window};
+int copy_batch = {new_window};
 bool should_overlap(int cached_tokens, int prompt_tokens) {{
-    return cached_tokens > prompt_tokens / 2;
+    return cached_tokens > (prompt_tokens * 4) / 5;
 }}
```

Request: identify the risk, give a minimal patch direction, and mention what benchmark evidence would prove the change worked.

"""

with open(transcript, "a", encoding="utf-8") as f:
    f.write(content)
    f.write(f"\\n\\nUser turn {turn}:\\n")
    f.write("Continue the optimization review. Focus on cache-aware prompt processing, scheduler overlap, and correctness gates.\\n\\n")
    for i in range(repeats):
        f.write(turn_unit.format(turn=turn, i=i, old_window=160 + turn + i, new_window=320 + turn + i))
    f.write("\\nAssistant:\\n")
PY
}

make_payload() {
    local prompt_file=$1
    local payload_file=$2
    python3 - "$prompt_file" "$payload_file" "$N_PREDICT" <<'PY'
import json
import sys

prompt_file, payload_file, n_predict = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(prompt_file, "r", encoding="utf-8") as f:
    prompt = f.read()

payload = {
    "prompt": prompt,
    "id_slot": 0,
    "n_predict": n_predict,
    "temperature": 0.0,
    "seed": 42,
    "stream": False,
    "cache_prompt": True,
    "timings_per_token": False,
}

with open(payload_file, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY
}

parse_turn() {
    local turn=$1
    local status=$2
    local prompt_file=$3
    local response_file=$4
    local slots_file=$5
    local wall_ms=$6
    python3 - "$turn" "$status" "$prompt_file" "$response_file" "$slots_file" "$wall_ms" "$RESULTS_TSV" <<'PY'
import hashlib
import json
import sys

turn, status, prompt_file, response_file, slots_file, wall_ms, results_tsv = sys.argv[1:]
turn_i = int(turn)

timings = {}
tokens_cached = ""
response_sha = ""
try:
    with open(response_file, "rb") as f:
        raw = f.read()
    response_sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    if data.get("error"):
        status = "response_error"
    timings = data.get("timings", {})
    tokens_cached = data.get("tokens_cached", "")
except Exception:
    if status == "ok":
        status = "response_parse_error"

cache_n = int(timings.get("cache_n", 0) or 0)
prompt_n = int(timings.get("prompt_n", 0) or 0)
predicted_n = int(timings.get("predicted_n", 0) or 0)
prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
predicted_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
processed_pp = float(timings.get("prompt_per_second", 0.0) or 0.0)
tg = float(timings.get("predicted_per_second", 0.0) or 0.0)
context_tokens = cache_n + prompt_n + predicted_n
denom = max(1, cache_n + prompt_n)
cache_ratio = cache_n / denom
effective_context_pp = ((cache_n + prompt_n) * 1000.0 / prompt_ms) if prompt_ms > 0 else 0.0

try:
    with open(slots_file, "r", encoding="utf-8") as f:
        slots = json.load(f)
    if slots and isinstance(slots, list):
        slot = slots[0]
        # Prefer API-visible slot depth after generation if present.
        context_tokens = max(context_tokens, int(slot.get("n_prompt_tokens", 0) or 0))
except Exception:
    pass

with open(results_tsv, "a", encoding="utf-8") as f:
    f.write(
        f"{turn_i}\t{status}\t{context_tokens}\t{cache_n}\t{prompt_n}\t{predicted_n}\t"
        f"{cache_ratio:.6f}\t{processed_pp:.6f}\t{effective_context_pp:.6f}\t{tg:.6f}\t"
        f"{prompt_ms:.3f}\t{predicted_ms:.3f}\t{wall_ms}\t{tokens_cached}\t"
        f"{response_sha}\t{prompt_file}\t{response_file}\t{slots_file}\n"
    )
PY
}

summarize_session() {
    python3 - "$RESULTS_TSV" "$SERVER_LOG" "$SUMMARY_JSON" "$SUMMARY_TXT" "$CTX" <<'PY'
import csv
import json
import re
import sys

results_tsv, server_log, summary_json, summary_txt, ctx = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = []
with open(results_tsv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["status"] == "ok":
            rows.append(row)

def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return 0.0

cached_rows = [r for r in rows if f(r, "cache_n") > 0 and int(r["turn"]) > 1]
primary_rows = cached_rows if cached_rows else rows[1:]
sum_prompt_ms = sum(f(r, "prompt_ms") for r in primary_rows)
sum_pred_ms = sum(f(r, "predicted_ms") for r in primary_rows)
sum_prompt = sum(f(r, "prompt_n") for r in primary_rows)
sum_cache = sum(f(r, "cache_n") for r in primary_rows)
sum_pred = sum(f(r, "predicted_n") for r in primary_rows)
max_context = max((f(r, "context_tokens") for r in rows), default=0.0)

log_text = ""
try:
    with open(server_log, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
except Exception:
    pass

f_keep = re.findall(r"f_keep = ([0-9.]+), sim = ([0-9.]+)", log_text)
cache_saves = len(re.findall(r"saving idle slot to prompt cache", log_text))
cache_loads = len(re.findall(r"loaded prompt from cache|found better prompt", log_text))
checkpoint_restores = len(re.findall(r"restored context checkpoint", log_text))

summary = {
    "results_tsv": results_tsv,
    "server_log": server_log,
    "turns_ok": len(rows),
    "primary_cached_turns": len(primary_rows),
    "max_context_tokens": int(max_context),
    "ctx_limit": ctx,
    "ctx_fill_ratio": max_context / ctx if ctx else 0.0,
    "aggregate_cached_turns": {
        "cache_tokens": int(sum_cache),
        "processed_prompt_tokens": int(sum_prompt),
        "predicted_tokens": int(sum_pred),
        "cache_ratio": (sum_cache / max(1.0, sum_cache + sum_prompt)),
        "processed_pp": (sum_prompt * 1000.0 / sum_prompt_ms) if sum_prompt_ms > 0 else 0.0,
        "effective_context_pp": ((sum_cache + sum_prompt) * 1000.0 / sum_prompt_ms) if sum_prompt_ms > 0 else 0.0,
        "tg": (sum_pred * 1000.0 / sum_pred_ms) if sum_pred_ms > 0 else 0.0,
        "prompt_ms": sum_prompt_ms,
        "predicted_ms": sum_pred_ms,
    },
    "reuse_log_evidence": {
        "f_keep_sim_pairs": f_keep[-10:],
        "prompt_cache_saves": cache_saves,
        "prompt_cache_load_or_better_prompt_events": cache_loads,
        "checkpoint_restores": checkpoint_restores,
    },
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("MiniMax cache-aware coding-session benchmark summary\n")
    f.write(f"turns_ok: {summary['turns_ok']}\n")
    f.write(f"primary_cached_turns: {summary['primary_cached_turns']}\n")
    f.write(f"max_context_tokens: {summary['max_context_tokens']} / {ctx} ({summary['ctx_fill_ratio']:.3f})\n")
    agg = summary["aggregate_cached_turns"]
    f.write(f"cache_ratio_cached_turns: {agg['cache_ratio']:.6f}\n")
    f.write(f"processed_pp_cached_turns: {agg['processed_pp']:.6f}\n")
    f.write(f"effective_context_pp_cached_turns: {agg['effective_context_pp']:.6f}\n")
    f.write(f"tg_cached_turns: {agg['tg']:.6f}\n")
    f.write(f"prompt_cache_saves: {cache_saves}\n")
    f.write(f"prompt_cache_load_or_better_prompt_events: {cache_loads}\n")
    f.write(f"checkpoint_restores: {checkpoint_restores}\n")
    if f_keep:
        f.write("recent_f_keep_sim_pairs: " + ", ".join(f"{a}/{b}" for a, b in f_keep[-5:]) + "\n")
PY
}

cmd=(
    "$BIN"
    -m "$MODEL"
    -c "$CTX"
    -ngl 999
    -sm layer
    -dev "$DEV"
    -fa on
    -np 1
    -fit off
    --numa distribute
    --no-warmup
    --jinja
    -t 24
    -tb 48
    -lv 3
    -ctk q4_0
    -ctv q4_0
    -b "$BATCH"
    -ub "$UBATCH"
    --placement-policy "$PLACEMENT_POLICY"
    --cuda-fattn-no-stream-k
    --cuda-fattn-vec-qkv-headroom-mib 0
    --reserve-pp-outputs 1
    --cuda-fattn-mma-q4-0-direct
    --cuda-fattn-mma-q4-0-direct-mode all
    --cache-prompt
    --cache-ram "$CACHE_RAM_MIB"
    --host 127.0.0.1
    --port "$PORT"
)

if [[ -n "$DEVICE_SPEED" ]]; then
    cmd+=(--device-speed "$DEVICE_SPEED")
fi

if [[ -n "$LAYER_SPLIT" ]]; then
    cmd+=(--layer-split "$LAYER_SPLIT")
fi

if [[ -n "$OUTPUT_DEVICE" ]]; then
    cmd+=(--output-device "$OUTPUT_DEVICE")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=($EXTRA_ARGS)
    cmd+=("${extra_args_array[@]}")
fi

printf '%q ' "${cmd[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"

"${cmd[@]}" > "$SERVER_LOG" 2>&1 &
srv_pid=$!
trap 'cleanup_server "$srv_pid"' EXIT
sample_telemetry "$srv_pid" &
telemetry_pid=$!

if ! wait_for_server "$srv_pid"; then
    echo "server failed to start; see $SERVER_LOG" >&2
    cleanup_server "$srv_pid"
    cleanup_server "$telemetry_pid"
    exit 1
fi

for turn in $(seq 1 "$MAX_TURNS"); do
    prompt_file="$RUN_DIR/turn-${turn}.prompt.txt"
    payload_file="$RUN_DIR/turn-${turn}.payload.json"
    response_file="$RUN_DIR/turn-${turn}.response.json"
    slots_file="$RUN_DIR/turn-${turn}.slots.json"
    curl_err="$RUN_DIR/turn-${turn}.curl.err"
    cp "$TRANSCRIPT" "$prompt_file"
    make_payload "$prompt_file" "$payload_file"

    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] turn $turn request"
    start_ns=$(date +%s%N)
    curl_status=0
    curl -sS --fail-with-body --max-time 3600 \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload_file" \
        "http://127.0.0.1:${PORT}/completion" > "$response_file" 2>"$curl_err" || curl_status=$?
    end_ns=$(date +%s%N)
    wall_ms=$(( (end_ns - start_ns) / 1000000 ))

    curl -sS "http://127.0.0.1:${PORT}/slots" > "$slots_file" 2>/dev/null || true

    if [[ $curl_status -eq 0 ]]; then
        parse_turn "$turn" "ok" "$prompt_file" "$response_file" "$slots_file" "$wall_ms"
    else
        parse_turn "$turn" "curl_failed_$curl_status" "$prompt_file" "$response_file" "$slots_file" "$wall_ms"
        break
    fi

    context_tokens=$(tail -n 1 "$RESULTS_TSV" | awk -F'\t' '{print $3}')
    cache_n=$(tail -n 1 "$RESULTS_TSV" | awk -F'\t' '{print $4}')
    prompt_n=$(tail -n 1 "$RESULTS_TSV" | awk -F'\t' '{print $5}')
    processed_pp=$(tail -n 1 "$RESULTS_TSV" | awk -F'\t' '{print $8}')
    echo "turn $turn context_tokens=$context_tokens cache_n=$cache_n prompt_n=$prompt_n processed_pp=$processed_pp"

    if [[ "${context_tokens:-0}" =~ ^[0-9]+$ ]] && (( context_tokens >= TARGET_CONTEXT )); then
        break
    fi
    if python3 - "$processed_pp" "$PP_STOP_CUTOFF" <<'PY'
import sys
pp = float(sys.argv[1] or 0)
cutoff = float(sys.argv[2])
raise SystemExit(0 if pp < cutoff else 1)
PY
    then
        echo "stopping: processed_pp=$processed_pp below PP_STOP_CUTOFF=$PP_STOP_CUTOFF"
        break
    fi
    next_turn=$((turn + 1))
    append_next_turn "$next_turn" "$response_file"
done

summarize_session
cleanup_server "$srv_pid"
cleanup_server "$telemetry_pid"
trap - EXIT

echo "summary: $SUMMARY_TXT"
cat "$SUMMARY_TXT"
