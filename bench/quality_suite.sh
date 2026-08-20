#!/bin/bash
# HumanEval + GSM8K quality suite: fusion OFF vs TideSpec v4.6 ON.
# Server runs with --default-chat-template-kwargs reasoning_effort=low so the
# stock run_eval clients inherit the operating condition.
set -u
# Host-side huggingface cache dir (override with env var if it lives elsewhere)
HF_CACHE="${HF_CACHE:-$USERPROFILE/.cache/huggingface}"

run_phase() {
  NAME=$1; ENVV=$2; IMG=$3
  docker rm -f ab-server >/dev/null 2>&1
  MSYS_NO_PATHCONV=1 docker run -d --name ab-server --gpus all --shm-size 32g -p 30000:30000 \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    --env SGLANG_USE_CUDA_IPC_TRANSPORT=0 --env SGLANG_DFLASH_NGRAM_FUSION=$ENVV --ipc=host \
    "$IMG" sglang serve --trust-remote-code \
    --model-path RadixArk/Qwen3.8-27B-NVFP4 --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4 \
    --default-chat-template-kwargs '{"reasoning_effort": "low"}' \
    --mem-fraction-static 0.93 --attention-backend flashinfer --max-running-requests 1 \
    --cuda-graph-max-bs-decode 1 --chunked-prefill-size 2048 --context-length 32768 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
    --speculative-draft-model-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
    --speculative-num-draft-tokens 8 --speculative-draft-model-quantization unquant \
    --weight-loader-disable-mmap --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy --kv-cache-dtype fp8_e4m3 \
    --host 0.0.0.0 --port 30000 >/dev/null || { echo "[$NAME] docker run FAILED"; return 1; }
  for i in $(seq 1 120); do
    STATE=$(docker inspect ab-server --format '{{.State.Running}}' 2>/dev/null || echo gone)
    [ "$STATE" != "true" ] && { echo "[$NAME] SERVER DIED:"; docker logs ab-server 2>&1 | tail -4; return 1; }
    docker logs ab-server 2>&1 | grep -q "Uvicorn running" && break
    sleep 5
  done
  docker logs ab-server 2>&1 | grep -q "Uvicorn running" || { echo "[$NAME] START TIMEOUT"; return 1; }
  docker logs ab-server 2>&1 | grep -q "will attempt download" && { echo "[$NAME] MOUNT BROKEN"; return 1; }
  sleep 8

  echo "=== [$NAME] HumanEval (164, pass@1 greedy 1-sample, temp 0) ==="
  docker run --rm --network container:ab-server --entrypoint bash "$IMG" -c "
    pip install -q human-eval 'filelock<3.18' 2>/dev/null;
    F=\$(python3 -c 'import human_eval.execution as e; print(e.__file__)');
    sed -i 's|#[[:space:]]*exec(check_program, exec_globals)|exec(check_program, exec_globals)|' \"\$F\";
    sed -i 's/num_samples_per_task: int = 5/num_samples_per_task: int = 1/' /sgl-workspace/sglang/python/sglang/test/simple_eval_humaneval.py;
    python3 -u -m sglang.test.run_eval --host 127.0.0.1 --port 30000 \
      --model RadixArk/Qwen3.8-27B-NVFP4 --eval-name humaneval \
      --max-tokens 1300 --temperature 0 --num-threads 2 2>&1 | grep -iE 'pass@|score|accept length' | tail -4"

  echo "=== [$NAME] GSM8K (200, temp 0) ==="
  docker run --rm --network container:ab-server --entrypoint python3 "$IMG" \
    -u -m sglang.test.run_eval --host 127.0.0.1 --port 30000 \
    --model RadixArk/Qwen3.8-27B-NVFP4 --eval-name gsm8k --num-examples 200 \
    --max-tokens 900 --temperature 0 --num-threads 2 2>&1 | grep -iE 'pass@|score|accept length' | tail -4

  docker logs ab-server 2>&1 | grep -oE "ngram-fusion v4.*streak=[0-9]+" | tail -1
  docker rm -f ab-server >/dev/null 2>&1
}

run_phase "fusion-OFF" 0 sglang:dflash2-fusion
run_phase "TideSpec" 1 sglang:dflash2-fusion
echo QUALITY-SUITE-DONE
