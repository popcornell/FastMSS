# TTS pipeline setup

The TTS recipes (`recipes/tts_staged.py`, `recipes/tts_sim.py`) generate meetings whose speech is synthesised rather than
concatenated from a corpus: an HMM samples the turn-taking skeleton, an LLM fills
it with dialogue, CosyVoice 3 renders each line cloning a LibriSpeech reference
clip, and the usual FastMSS mixing/RIR path takes it from there.

Three things live outside this repo and must be set up first. Each is either
found automatically or pointed at with an environment variable.

## 1. CosyVoice 3

CosyVoice pins `torch 2.3.1` / `numpy 1.26` / `transformers 4.51`, which would
downgrade FastMSS's environment. So it is **not** a dependency: it runs as a
persistent subprocess in its own conda env, and `fastmss/tts/backends.py` talks
to it over a JSON-line pipe (`fastmss/tts/cosyvoice_worker.py` is the other end,
and does live in this repo).

Clone it and fetch the weights:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice third_party/CosyVoice
export COSYVOICE_REPO=$PWD/third_party/CosyVoice

# Fun-CosyVoice3-0.5B into $COSYVOICE_REPO/pretrained_models/
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B \
    --local-dir $COSYVOICE_REPO/pretrained_models/Fun-CosyVoice3-0.5B
```

Build its env:

```bash
conda create -n cosyvoice python=3.10 -y && conda activate cosyvoice
conda install -c conda-forge pynini=2.1.6 -y
pip install -r $COSYVOICE_REPO/requirements.txt
export COSYVOICE_PYTHON=$(python -c 'import sys; print(sys.executable)')
conda deactivate
```

| variable | default if unset |
| --- | --- |
| `COSYVOICE_REPO` | `~/CosyVoice`, else nearest `third_party/CosyVoice` walking up from the package |
| `COSYVOICE_PYTHON` | `<conda root>/envs/cosyvoice/bin/python` |
| `COSYVOICE_MODEL` | `$COSYVOICE_REPO/pretrained_models/Fun-CosyVoice3-0.5B` |

The worker uses whatever GPU `CUDA_VISIBLE_DEVICES` exposes and falls back to
CPU otherwise. The model loads once per run, not once per utterance, and
prompt-derived speaker features are memoised per reference clip.

Acceleration is opt-in, because it depends on how the local CosyVoice env was
built. `recipes/tts_staged.py` sets these from its `--cosyvoice-*` flags:

| variable | flag | effect |
| --- | --- | --- |
| `COSYVOICE_FP16=1` | `--cosyvoice-fp16` | fp16 inference (CUDA only) |
| `COSYVOICE_LOAD_TRT=1` | `--cosyvoice-load-trt` | TensorRT flow decoder; the first run builds a `.plan` engine |
| `COSYVOICE_TRT_CONCURRENT=N` | `--cosyvoice-trt-concurrent N` | TensorRT context count |
| `COSYVOICE_LOAD_VLLM=1` | `--cosyvoice-load-vllm` | vLLM backend; needs `vllm` in the env and a `vllm/` model export |
| `COSYVOICE_TEXT_FRONTEND=0` | `--cosyvoice-no-text-frontend` | skip CosyVoice text normalisation |

## 2. LibriSpeech

Speaker identity is a reference clip, so the voice bank is built from
LibriSpeech (`fastmss/tts/refbank.py`). Pass `--librispeech`, or:

```bash
export LIBRISPEECH_ROOT=/path/to/LibriSpeech
```

## 3. An LLM endpoint

Dialogue fill (`fastmss/tts/fill.py`) needs a chat endpoint with JSON-schema
output. The default is a local Ollama server:

```bash
ollama serve && ollama pull qwen3:30b-a3b
```

A local Ollama URL (port 11434) goes to Ollama's native `/api/chat`; anything
else is treated as an OpenAI-compatible `/v1` server (vLLM, SGLang, ...). Set
`FASTMSS_LLM_API=openai` to force the OpenAI route. Override the endpoint with
`--model` / `--base-url`, or skip the LLM entirely with `--no-llm`, which stubs
the dialogue and still runs the timing + TTS + mixing path.

| variable | default | meaning |
| --- | --- | --- |
| `FASTMSS_LLM_MAX_TOKENS` | 1200 | default output cap (`--llm-max-tokens`) |
| `FASTMSS_LLM_CAST_MAX_TOKENS` | max(900, 260 x speakers) | premise + personas call (`--llm-cast-max-tokens`) |
| `FASTMSS_LLM_FILL_MAX_TOKENS` | 1600 | upper bound per dialogue chunk (`--llm-fill-max-tokens`) |
| `FASTMSS_LLM_TEMPERATURE` | 0.35 | sampling temperature |
| `FASTMSS_LLM_TOP_P` | 0.9 | nucleus sampling |

If long meetings fail with "LLM output hit max_tokens", raise the matching
budget or lower `--fill-chunk`.

## Smoke tests

```bash
# is the worker env sane? should print {"ready": true, "sample_rate": 24000}
echo '' | $COSYVOICE_PYTHON -u fastmss/tts/cosyvoice_worker.py

# timing + TTS + mixing without an LLM
python recipes/tts_staged.py --phase all --no-llm --n 1 --duration 15 --out exp/smoke
```

The run commands and every recipe option are described in the
[README](../README.md).
