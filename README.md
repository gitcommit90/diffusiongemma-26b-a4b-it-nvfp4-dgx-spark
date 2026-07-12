# DiffusionGemma 26B A4B IT NVFP4 — DGX Spark / GB10

Ready-to-run **vLLM deployment package** for [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) on **NVIDIA GB10 (DGX Spark)**.

This is a **deployment package**, not new weights. Upstream checkpoint: NVIDIA ModelOpt NVFP4 of Google DeepMind DiffusionGemma 26B A4B IT (MoE, 25.2B total / ~3.8B active). Runtime image: `vllm/vllm-openai:gemma-aarch64-cu130`.

## Hardware measured on

| Item | Value |
|------|--------|
| Host | ASUS Ascent GX10 (DGX Spark class) |
| GPU | NVIDIA GB10 |
| Unified memory | ~121 GiB |
| OS | Ubuntu aarch64, CUDA 13.0 |
| Driver | 580.159.03 |

## What you get

- `./start.sh` — download model if missing, launch vLLM with **10 concurrent sequences**
- `./stop.sh` — stop/remove container
- `bench_concurrent.py` + `bench_results.json` — streaming concurrent bench (1 / 4 / 10)
- Official gemma-image flags: `TRITON_ATTN`, gemma4 tool/reasoning parsers, thinking enabled

## Quick start

```bash
# optional overrides
export MODEL_DIR=$HOME/llm/diffusiongemma-26b-a4b-it-nvfp4
export PORT=8000
export MAX_NUM_SEQS=10

./start.sh
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
  "messages": [{"role":"user","content":"In one short sentence, what is NVFP4?"}],
  "max_tokens": 256,
  "temperature": 0.3,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
```

With thinking enabled (default serve config), final answer lands in `message.content` and chain-of-thought in `message.reasoning`. For short answers without thinking, pass `chat_template_kwargs.enable_thinking=false` as above.

## Exact serve command (measured config)

Image digest used on this ship: `vllm/vllm-openai@sha256:9c719fc0c869092c7d0533f8357d6985a38d5ff03b20ffb6a4620c2b4806dd4b`  
(`vllm/vllm-openai:gemma-aarch64-cu130` / multi-arch `gemma` tag on aarch64)

```bash
docker run -d --name diffusiongemma-26b-nvfp4 \
  --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -v $HOME/llm/diffusiongemma-26b-a4b-it-nvfp4:/models/diffusiongemma:ro \
  vllm/vllm-openai:gemma-aarch64-cu130 \
  /models/diffusiongemma \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-num-seqs 10 \
    --attention-backend TRITON_ATTN \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --override-generation-config '{"max_new_tokens": null}' \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --served-model-name nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768
```

**Note:** image `ENTRYPOINT` is already `["vllm","serve"]` — pass the model path as the first arg, **not** a second `serve`.

## Measured benchmarks (this machine)

Streaming chat completions, `max_tokens=256`, thinking **on** (default), prompt about NVFP4. TTFT = first streamed token (content **or** reasoning). Warm single-stream is the practical headline.

| Concurrency | Succeeded | Avg TTFT | Aggregate tok/s | Per-stream tok/s |
|------------:|----------:|---------:|----------------:|-----------------:|
| 1 | 1/1 | **1.70 s** | **150.7** | 150.7 |
| 4 | 4/4 | 3.67 s | **264.5** | 69.9 |
| 10 | **10/10** | 14.91 s | **149.0** | 17.3 |

Source: `bench_results.json` (2026-07-12).

### Coherence sample (thinking off)

> NVFP4 is a specialized neural network quantization technique that compresses model weights into a 4-bit floating-point format to reduce model size while maintaining high accuracy.

### Concurrent status

**10 concurrent achieved and stable** (`--max-num-seqs 10`, all 10 workers completed). Peak measured aggregate throughput on this run was at **concurrency=4** (~265 tok/s).

## Model notes

- Architecture: `DiffusionGemmaForBlockDiffusion` (discrete diffusion / block generation, not classic AR-only)
- Quant: ModelOpt **NVFP4** (~18.9 GB on disk)
- vLLM auto-selected `modelopt_fp4` + FlashInfer CUTLASS NVFP4 MoE backend
- KV cache: fp8_e4m3 (from quant scheme)
- Context: model supports up to 256K; this package defaults to **32K** for practical GB10 headroom — raise `MAX_MODEL_LEN` if you have room
- Official NVIDIA recipe used `max_num_seqs 4`; this package targets **10** and measured it

## Reproduce bench

```bash
python3 bench_concurrent.py \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --levels 1,4,10 \
  --max-tokens 256 \
  --warmup 1 \
  --out bench_results.json
```

## Credits

- Weights: [nvidia/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) (base: google/diffusiongemma-26B-A4B-it)
- Runtime: [vLLM](https://github.com/vllm-project/vllm) `vllm/vllm-openai:gemma` family
- License: Apache 2.0 + Gemma Terms of Use

## Caveats

1. Diffusion / block generation can make TTFT look like “full block” latency more than classic first-token AR — numbers are still honest wall-clock from stream start to first delta.
2. With thinking enabled, short `max_tokens` may fill with `reasoning` only; raise `max_tokens` or set `enable_thinking: false` for short final answers.
3. Disk ~19G for weights; do not commit safetensors to git.
4. JSON file paths for `--override-generation-config` are **not** accepted by this image — pass inline JSON strings.
