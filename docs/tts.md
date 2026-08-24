# TTS meeting simulation

`recipes/tts_sim.py` generates meetings whose speech is synthesised rather than
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
| `COSYVOICE_REPO` | nearest `third_party/CosyVoice` walking up from the package |
| `COSYVOICE_PYTHON` | `<conda root>/envs/cosyvoice/bin/python` |
| `COSYVOICE_MODEL` | `$COSYVOICE_REPO/pretrained_models/Fun-CosyVoice3-0.5B` |

The worker runs on CPU: fp16 is CUDA-only in that codebase and MPS support is
partial. It is a 0.5B model, so this is workable but not fast; the model loads
once per run, not once per utterance, and prompt-derived speaker features are
memoised per reference clip.

## 2. LibriSpeech

Speaker identity is a reference clip, so the voice bank is built from
LibriSpeech (`fastmss/tts/refbank.py`). Pass `--librispeech`, or:

```bash
export LIBRISPEECH_ROOT=/path/to/LibriSpeech
```

## 3. An OpenAI-compatible LLM endpoint

Dialogue fill (`fastmss/tts/fill.py`) posts to an OpenAI-compatible `/v1`
endpoint — an Ollama server by default:

```bash
ollama serve && ollama pull qwen3:30b-a3b
```

Override with `--model` / `--base-url`, or skip it entirely with `--no-llm`,
which stubs the fill and still exercises the timing + TTS + mixing path.

## Run

```bash
python recipes/tts_sim.py --n 3 --duration 30 --speakers 3 --reverberate \
    --out exp/tts_sim
```

Outputs land in `--out`: `audio/`, `manifests/` (lhotse recording +
supervision sets), RTTMs, and `clips_native/` — the TTS cache, keyed by
(text, instruction, reference clip), so re-runs do not re-synthesise.

Smoke-test the pieces separately if something fails:

```bash
# is the worker env sane? should print {"ready": true, "sample_rate": 24000}
echo '' | $COSYVOICE_PYTHON -u fastmss/tts/cosyvoice_worker.py
```
