#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/stefan/llama.cpp-gfx906}
BIN=${BIN:-$ROOT/build-cuda-hip-dl-gfx906/bin/llama-server}
MODEL=${MODEL:-/home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf}
DEV=${DEV:-ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2}
CTX=${CTX:-69376}
N_PREDICT=${N_PREDICT:-128}
PROMPT_REPEATS=${PROMPT_REPEATS:-320}
PORT_BASE=${PORT_BASE:-18080}
RUN_DIR=${RUN_DIR:-$ROOT/bench-results/minimax-placement-$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$RUN_DIR"

PROMPT_FILE="$RUN_DIR/prompt.txt"
RESULTS_TSV="$RUN_DIR/results.tsv"

python3 - "$PROMPT_REPEATS" > "$PROMPT_FILE" <<'PY'
import sys

repeats = int(sys.argv[1])
unit = (
    "Benchmark paragraph: evaluate heterogeneous layer placement, CUDA waiting time, "
    "ROCm pipeline balance, KV cache residency, prompt processing throughput, token "
    "generation throughput, PCIe pressure, and output sanity. "
)
print(unit * repeats)
PY

printf "name\tstatus\tprompt_n\tpredicted_n\tprompt_per_second\tpredicted_per_second\tprompt_ms\tpredicted_ms\ttotal_ms\tload_seconds\tgenerated_split\tserver_log\tresponse_file\n" > "$RESULTS_TSV"

runs=(
    "baseline_q8|q8_0|q8_0|64|32|"
    "baseline_q4|q4_0|q4_0|128|64|"
    "auto_q8_max_cuda|q8_0|q8_0|64|32|--placement-policy max-cuda"
    "auto_q4_max_cuda|q4_0|q4_0|128|64|--placement-policy max-cuda"
    "auto_q4_max_cuda_b256_ub256|q4_0|q4_0|256|256|--placement-policy max-cuda"
    "auto_q4_balanced_default_b256_ub256|q4_0|q4_0|256|256|--placement-policy balanced"
    "auto_q4_balanced_measured_b256_ub256|q4_0|q4_0|256|256|--placement-policy balanced --device-speed 0.55,0.55,1.0,1.0,1.0,1.0"
    "auto_q4_balanced_measured_b416_ub416|q4_0|q4_0|416|416|--placement-policy balanced --device-speed 0.55,0.55,1.0,1.0,1.0,1.0"
    "auto_q8_balanced_measured|q8_0|q8_0|64|32|--placement-policy balanced --device-speed 0.55,0.55,1.0,1.0,1.0,1.0"
    "ts_q8_14_14_6_11_11_7|q8_0|q8_0|64|32|-ts 14,14,6,11,11,7"
    "ts_q8_15_14_6_11_11_6|q8_0|q8_0|64|32|-ts 15,14,6,11,11,6"
    "ts_q4_13_13_6_12_12_9|q4_0|q4_0|128|64|-ts 13,13,6,12,12,9"
    "layer_q8_14_14_6_11_11_6_out_cuda0|q8_0|q8_0|64|32|--layer-split 14,14,6,11,11,6 --output-device CUDA0"
    "layer_q8_15_14_6_10_11_6_out_cuda0|q8_0|q8_0|64|32|--layer-split 15,14,6,10,11,6 --output-device CUDA0"
    "layer_q4_13_13_6_12_12_6_out_cuda0|q4_0|q4_0|128|64|--layer-split 13,13,6,12,12,6 --output-device CUDA0"
)

record_failure() {
    local name=$1
    local status=$2
    local log_file=$3
    local response_file=$4
    local load_seconds=${5:-}
    printf "%s\t%s\t\t\t\t\t\t\t\t%s\t\t%s\t%s\n" "$name" "$status" "$load_seconds" "$log_file" "$response_file" >> "$RESULTS_TSV"
}

sample_telemetry() {
    local pid=$1
    local out=$2
    while kill -0 "$pid" 2>/dev/null; do
        {
            date -u '+=== %Y-%m-%dT%H:%M:%SZ ==='
            ps -o pid,rss,vsz,pcpu,pmem,comm -p "$pid" 2>/dev/null || true
            nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null || true
            rocm-smi --showmeminfo vram --showuse --showbus 2>/dev/null || true
        } >> "$out"
        sleep 2
    done
}

wait_for_server() {
    local port=$1
    local pid=$2
    local log_file=$3
    for _ in $(seq 1 900); do
        if grep -q "server is listening" "$log_file" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if grep -qiE "failed to load|error loading model|out of memory|unable to allocate|failed to allocate" "$log_file" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
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
    "n_predict": n_predict,
    "temperature": 0.0,
    "seed": 42,
    "stream": False,
    "cache_prompt": False,
    "timings_per_token": False,
}

with open(payload_file, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY
}

parse_response() {
    local name=$1
    local status=$2
    local log_file=$3
    local response_file=$4
    local load_seconds=$5
    python3 - "$name" "$status" "$log_file" "$response_file" "$load_seconds" "$RESULTS_TSV" <<'PY'
import json
import sys

name, status, log_file, response_file, load_seconds, results_tsv = sys.argv[1:]

prompt_n = predicted_n = prompt_ps = predicted_ps = prompt_ms = predicted_ms = total_ms = ""
generated_split = ""
try:
    import re
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    m = re.search(r"placement_policy=(?:max-cuda|balanced) generated layer split ([0-9,]+)", log_text)
    if m:
        generated_split = m.group(1)
except Exception:
    pass
try:
    with open(response_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    timings = data.get("timings", {})
    prompt_n = timings.get("prompt_n", "")
    predicted_n = timings.get("predicted_n", "")
    prompt_ps = timings.get("prompt_per_second", "")
    predicted_ps = timings.get("predicted_per_second", "")
    prompt_ms = timings.get("prompt_ms", "")
    predicted_ms = timings.get("predicted_ms", "")
    total_ms = timings.get("total_ms", "")
    if data.get("error"):
        status = "response_error"
except Exception:
    status = "response_parse_error"

with open(results_tsv, "a", encoding="utf-8") as f:
    f.write(
        f"{name}\t{status}\t{prompt_n}\t{predicted_n}\t{prompt_ps}\t{predicted_ps}\t"
        f"{prompt_ms}\t{predicted_ms}\t{total_ms}\t{load_seconds}\t{generated_split}\t{log_file}\t{response_file}\n"
    )
PY
}

summarize_log() {
    local log_file=$1
    local summary_file=$2
    {
        rg -n "placement_policy=(max-cuda|balanced)|layer placement summary|backend=.*layers=|model loaded|offloading|KV self size|CPU buffer size|fallback|failed|error|out of memory|unable to allocate|failed to allocate|assigned to device|load_tensors:" "$log_file" || true
    } > "$summary_file"
}

cleanup_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
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

payload_file="$RUN_DIR/payload.json"
make_payload "$PROMPT_FILE" "$payload_file"

for i in "${!runs[@]}"; do
    IFS='|' read -r name ctk ctv batch ubatch extra <<< "${runs[$i]}"
    port=$((PORT_BASE + i))
    log_file="$RUN_DIR/${name}.server.log"
    telemetry_file="$RUN_DIR/${name}.telemetry.log"
    response_file="$RUN_DIR/${name}.response.json"
    log_summary_file="$RUN_DIR/${name}.log-summary.txt"
    cmd_file="$RUN_DIR/${name}.cmd.txt"

    read -r -a extra_args <<< "$extra"
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
        -ctk "$ctk"
        -ctv "$ctv"
        -b "$batch"
        -ub "$ubatch"
        --host 127.0.0.1
        --port "$port"
    )
    if [[ -n "$extra" ]]; then
        cmd+=("${extra_args[@]}")
    fi

    printf '%q ' "${cmd[@]}" > "$cmd_file"
    printf '\n' >> "$cmd_file"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting $name"

    start_epoch=$(date +%s)
    "${cmd[@]}" > "$log_file" 2>&1 &
    srv_pid=$!
    sample_telemetry "$srv_pid" "$telemetry_file" &
    telemetry_pid=$!

    if ! wait_for_server "$port" "$srv_pid" "$log_file"; then
        load_seconds=$(( $(date +%s) - start_epoch ))
        record_failure "$name" "server_start_failed" "$log_file" "$response_file" "$load_seconds"
        summarize_log "$log_file" "$log_summary_file"
        cleanup_server "$srv_pid"
        cleanup_server "$telemetry_pid"
        continue
    fi

    load_seconds=$(( $(date +%s) - start_epoch ))
    curl -sS --fail-with-body --max-time 3600 \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload_file" \
        "http://127.0.0.1:${port}/completion" > "$response_file" 2>"$RUN_DIR/${name}.curl.err"
    curl_status=$?

    if [[ $curl_status -eq 0 ]]; then
        parse_response "$name" "ok" "$log_file" "$response_file" "$load_seconds"
    else
        record_failure "$name" "curl_failed_$curl_status" "$log_file" "$response_file" "$load_seconds"
    fi

    curl -sS "http://127.0.0.1:${port}/slots" > "$RUN_DIR/${name}.slots.json" 2>/dev/null || true
    summarize_log "$log_file" "$log_summary_file"
    cleanup_server "$srv_pid"
    cleanup_server "$telemetry_pid"
    sleep 5
done

echo "results: $RESULTS_TSV"
