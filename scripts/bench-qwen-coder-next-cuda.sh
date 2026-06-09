#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/stefan/llama.cpp-gfx906}
BIN=${BIN:-$ROOT/build-cuda-hip-dl-gfx906/bin/llama-server}
MODEL=${MODEL:-/home/stefan/.lmstudio/models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf}
DEV=${DEV:-CUDA0,CUDA2,CUDA1,CUDA3}
TS=${TS:-13,9,13,9}
CTX=${CTX:-69376}
PORT_BASE=${PORT_BASE:-21840}
RUN_DIR=${RUN_DIR:-$ROOT/bench-results/qwen3-coder-next-cuda-q5-q8kv-$(date -u +%Y%m%dT%H%M%SZ)}
PAYLOAD_SRC=${PAYLOAD_SRC:-$ROOT/bench-results/qwen3-coder-next-cuda-q8kv-partial-20260605T014650Z/payload.json}
UBATCH=${UBATCH:-512}

mkdir -p "$RUN_DIR"
cp "$PAYLOAD_SRC" "$RUN_DIR/payload.json"

RESULTS_TSV="$RUN_DIR/results.tsv"
if [[ ! -f "$RESULTS_TSV" ]]; then
    printf "name\tstatus\tprompt_n\tpredicted_n\tprompt_per_second\tpredicted_per_second\tprompt_ms\tpredicted_ms\tcontent_sha256\tcontent_prefix\tserver_log\tresponse_file\n" > "$RESULTS_TSV"
fi

runs=(
    "qwen_q5_cuda4_q8kv_ctx69376_ngl999_gpuout_ub512|999|"
    "qwen_q5_cuda4_q8kv_ctx69376_ngl999_cpuout_ub512|999|--output-device CPU"
    "qwen_q5_cuda4_q8kv_ctx69376_ngl48_cpuout_ub512|48|--output-device CPU"
    "qwen_q5_cuda4_q8kv_ctx69376_ngl47_cpuout_ub512|47|--output-device CPU"
    "qwen_q5_cuda4_q8kv_ctx69376_ngl46_cpuout_ub512|46|--output-device CPU"
    "qwen_q5_cuda4_q8kv_ctx69376_ngl45_cpuout_ub512|45|--output-device CPU"
    "qwen_q5_cuda4_q8kv_ctx69376_ngl44_cpuout_ub512|44|--output-device CPU"
)

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

wait_for_server() {
    local pid=$1
    local log_file=$2
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

record_failure() {
    local name=$1
    local status=$2
    local log_file=$3
    local response_file=$4
    printf "%s\t%s\t\t\t\t\t\t\t\t\t%s\t%s\n" "$name" "$status" "$log_file" "$response_file" >> "$RESULTS_TSV"
}

parse_response() {
    local name=$1
    local status=$2
    local log_file=$3
    local response_file=$4
    python3 - "$name" "$status" "$log_file" "$response_file" "$RESULTS_TSV" <<'PY'
import hashlib
import json
import sys

name, status, log_file, response_file, results_tsv = sys.argv[1:]
prompt_n = predicted_n = prompt_ps = predicted_ps = prompt_ms = predicted_ms = ""
content_sha = content_prefix = ""
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
    content = data.get("content", "")
    content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else ""
    content_prefix = content[:120].replace("\n", "\\n").replace("\t", " ")
    if data.get("error"):
        status = "response_error"
except Exception:
    status = "response_parse_error"

with open(results_tsv, "a", encoding="utf-8") as f:
    f.write(
        f"{name}\t{status}\t{prompt_n}\t{predicted_n}\t{prompt_ps}\t{predicted_ps}\t"
        f"{prompt_ms}\t{predicted_ms}\t{content_sha}\t{content_prefix}\t{log_file}\t{response_file}\n"
    )
PY
}

summarize_log() {
    local log_file=$1
    local summary_file=$2
    rg -n "CUDA[0-9]|CPU|layer placement summary|backend=.*layers=|placement_policy=|out of memory|failed|fallback|sched_reserve|server is listening|prompt eval time|eval time" "$log_file" > "$summary_file" || true
}

if [[ ! -f "$MODEL" ]]; then
    echo "model file not found: $MODEL" >&2
    exit 1
fi

if [[ "$MODEL" =~ ^(.*)-00001-of-([0-9]{5})\.gguf$ ]]; then
    prefix="${BASH_REMATCH[1]}"
    total="${BASH_REMATCH[2]}"
    for part in $(seq -f "%05g" 1 "$((10#$total))"); do
        split_path="${prefix}-${part}-of-${total}.gguf"
        if [[ ! -f "$split_path" ]]; then
            echo "split model is incomplete, missing: $split_path" >&2
            exit 1
        fi
    done
    shopt -s nullglob
    downloading_parts=("${MODEL%/*}"/downloading_"${prefix##*/}"-*-of-"${total}".gguf.part)
    shopt -u nullglob
    if (( ${#downloading_parts[@]} > 0 )); then
        echo "split model is still downloading:" >&2
        printf '  %s\n' "${downloading_parts[@]}" >&2
        exit 1
    fi
fi

for i in "${!runs[@]}"; do
    IFS='|' read -r name ngl extra <<< "${runs[$i]}"
    if [[ -n "${RUN_FILTER:-}" && "$name" != *"$RUN_FILTER"* ]]; then
        continue
    fi

    port=$((PORT_BASE + i))
    log_file="$RUN_DIR/${name}.server.log"
    response_file="$RUN_DIR/${name}.response.json"
    cmd_file="$RUN_DIR/${name}.cmd.txt"
    summary_file="$RUN_DIR/${name}.summary.txt"

    read -r -a extra_args <<< "$extra"
    read -r -a common_extra_args <<< "${COMMON_EXTRA_ARGS:-}"
    cmd=(
        "$BIN"
        -m "$MODEL"
        -c "$CTX"
        -ngl "$ngl"
        -sm layer
        -dev "$DEV"
        -ts "$TS"
        -fa on
        -np 1
        -fit off
        --numa distribute
        --no-warmup
        --jinja
        -t 24
        -tb 48
        -lv 3
        -ctk q8_0
        -ctv q8_0
        -b "$UBATCH"
        -ub "$UBATCH"
        --placement-policy memory
        --reserve-pp-outputs 1
        "${common_extra_args[@]}"
        --host 127.0.0.1
        --port "$port"
    )
    if [[ -n "$extra" ]]; then
        cmd+=("${extra_args[@]}")
    fi

    printf '%q ' "${cmd[@]}" > "$cmd_file"
    printf '\n' >> "$cmd_file"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting $name"

    "${cmd[@]}" > "$log_file" 2>&1 &
    srv_pid=$!
    if ! wait_for_server "$srv_pid" "$log_file"; then
        record_failure "$name" "server_start_failed" "$log_file" "$response_file"
        summarize_log "$log_file" "$summary_file"
        cleanup_server "$srv_pid"
        continue
    fi

    curl -sS --fail-with-body --max-time 3600 \
        -H 'Content-Type: application/json' \
        --data-binary "@$RUN_DIR/payload.json" \
        "http://127.0.0.1:${port}/completion" > "$response_file" 2>"$RUN_DIR/${name}.curl.err"
    curl_status=$?

    if [[ $curl_status -eq 0 ]]; then
        parse_response "$name" "ok" "$log_file" "$response_file"
    else
        record_failure "$name" "curl_failed_$curl_status" "$log_file" "$response_file"
    fi

    curl -sS "http://127.0.0.1:${port}/slots" > "$RUN_DIR/${name}.slots.json" 2>/dev/null || true
    summarize_log "$log_file" "$summary_file"
    cleanup_server "$srv_pid"

    if [[ $curl_status -eq 0 ]]; then
        break
    fi

    sleep 3
done

echo "$RUN_DIR"
cat "$RESULTS_TSV"
