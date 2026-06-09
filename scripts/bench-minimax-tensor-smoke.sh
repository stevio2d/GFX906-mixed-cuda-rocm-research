#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/stefan/llama.cpp-gfx906}
BIN=${BIN:-$ROOT/build-cuda-hip-dl-gfx906/bin/llama-server}
MODEL=${MODEL:-/home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf}
DEV=${DEV:-CUDA0,CUDA1,CUDA2,CUDA3,ROCm0,ROCm1}
TS=${TS:-33,33,17,17,48,52}
CTX=${CTX:-4096}
BATCH=${BATCH:-64}
UBATCH=${UBATCH:-$BATCH}
CACHE_K=${CACHE_K:-f16}
CACHE_V=${CACHE_V:-f16}
PORT=${PORT:-21690}
RUN_DIR=${RUN_DIR:-$ROOT/bench-results/minimax-tensor-smoke-f16kv-$(date -u +%Y%m%dT%H%M%SZ)}
PREDICT=${PREDICT:-16}
PAYLOAD_SRC=${PAYLOAD_SRC:-}
PROMPT_REPEATS=${PROMPT_REPEATS:-0}

export PATH=/opt/rocm/bin:${PATH:-}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.0.6}
export ROCBLAS_LAYER=${ROCBLAS_LAYER:-0}
export ROCBLAS_LOG_LEVEL=${ROCBLAS_LOG_LEVEL:-0}
export HIPBLASLT_LOG_LEVEL=${HIPBLASLT_LOG_LEVEL:-0}
export HIP_FORCE_DEV_KERNARG=${HIP_FORCE_DEV_KERNARG:-1}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-8}
export ROCBLAS_TENSILE_LIBPATH=${ROCBLAS_TENSILE_LIBPATH:-/opt/rocm-custom/lib/rocblas/library}
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}

export LLAMA_UNSAFE_ALLOW_UNSUPPORTED_TENSOR_SPLIT=${LLAMA_UNSAFE_ALLOW_UNSUPPORTED_TENSOR_SPLIT:-1}
export GGML_MIXED_VENDOR_COPY=${GGML_MIXED_VENDOR_COPY:-1}
export GGML_MIXED_VENDOR_COPY_CHUNK_BYTES=${GGML_MIXED_VENDOR_COPY_CHUNK_BYTES:-32M}
export GGML_MIXED_VENDOR_COPY_BUFFERS=${GGML_MIXED_VENDOR_COPY_BUFFERS:-8}
export GGML_META_ALLOC_LOG=${GGML_META_ALLOC_LOG:-1}
if [[ -n "${PLAN_ONLY:-}" ]]; then
    export GGML_META_ALLOC_PLAN_ONLY=1
fi

mkdir -p "$RUN_DIR"
if [[ -n "$PAYLOAD_SRC" ]]; then
    cp "$PAYLOAD_SRC" "$RUN_DIR/payload.json"
elif [[ "$PROMPT_REPEATS" != "0" ]]; then
    python3 - "$PROMPT_REPEATS" "$PREDICT" "$RUN_DIR/payload.json" <<'PY'
import json
import sys

repeats = int(sys.argv[1])
n_predict = int(sys.argv[2])
payload_file = sys.argv[3]
unit = (
    "Benchmark paragraph: evaluate heterogeneous tensor placement, CUDA collectives, "
    "ROCm stages, KV cache residency, prompt processing throughput, PCIe pressure, "
    "and output sanity. "
)
payload = {
    "prompt": unit * repeats,
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
else
    cat > "$RUN_DIR/payload.json" <<JSON
{"prompt":"Benchmark prompt: summarize one practical llama.cpp mixed GPU optimization.","n_predict":$PREDICT,"temperature":0.0,"seed":42,"stream":false,"cache_prompt":false,"timings_per_token":false}
JSON
fi

cmd=(
    "$BIN"
    -m "$MODEL"
    -c "$CTX"
    -ngl 999
    -sm tensor
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
    -ctk "$CACHE_K"
    -ctv "$CACHE_V"
    -b "$BATCH"
    -ub "$UBATCH"
    --host 127.0.0.1
    --port "$PORT"
)

printf '%q ' "${cmd[@]}" > "$RUN_DIR/cmd.txt"
printf '\n' >> "$RUN_DIR/cmd.txt"

"${cmd[@]}" > "$RUN_DIR/server.log" 2>&1 &
srv_pid=$!

run_status=server_timeout
for _ in $(seq 1 900); do
    if grep -q "server is listening" "$RUN_DIR/server.log" 2>/dev/null; then
        run_status=server_ready
        break
    fi
    if ! kill -0 "$srv_pid" 2>/dev/null; then
        run_status=server_exited
        break
    fi
    if grep -qiE "failed to load|error loading model|out of memory|unable to allocate|failed to allocate|GGML_ASSERT|SPLIT_MODE_TENSOR.*not" "$RUN_DIR/server.log" 2>/dev/null; then
        run_status=server_failed
        break
    fi
    sleep 2
done

if [[ "$run_status" == server_ready ]]; then
    curl -sS --fail-with-body --max-time 600 \
        -H 'Content-Type: application/json' \
        --data-binary "@$RUN_DIR/payload.json" \
        "http://127.0.0.1:$PORT/completion" > "$RUN_DIR/response.json" 2>"$RUN_DIR/curl.err" || run_status=curl_failed
    curl -sS "http://127.0.0.1:$PORT/slots" > "$RUN_DIR/slots.json" 2>/dev/null || true
fi

if kill -0 "$srv_pid" 2>/dev/null; then
    kill "$srv_pid" 2>/dev/null || true
    sleep 2
fi
if kill -0 "$srv_pid" 2>/dev/null; then
    kill -9 "$srv_pid" 2>/dev/null || true
fi

python3 - "$RUN_DIR" "$run_status" <<'PY'
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
status = sys.argv[2]
print(run)
print("status", status)
r = run / "response.json"
if r.exists():
    data = json.load(open(r, encoding="utf-8"))
    timings = data.get("timings", {})
    print("prompt_n", timings.get("prompt_n"), "predicted_n", timings.get("predicted_n"))
    print("pp", timings.get("prompt_per_second"), "tg", timings.get("predicted_per_second"))
    content = data.get("content", "")
    print("sha", hashlib.sha256(content.encode()).hexdigest() if content else "")
    print("content", repr(content[:160]))
PY

rg -n "simple [0-9].*planned|failed to allocate simple|creating a Meta device|device [0-9]:|allocating|failed|GGML_ASSERT|server is listening|mul_mat_id|SPLIT_MODE_TENSOR|prompt eval time|eval time" "$RUN_DIR/server.log" -S || true
