"""Persistent CosyVoice 3 worker, run inside its own conda env.

CosyVoice pins torch 2.3.1 / numpy 1.26 / transformers 4.51, which would downgrade the
pipeline's environment. So it lives in a separate env (`conda create -n cosyvoice
python=3.10`) and the pipeline talks to it over a pipe instead of importing it:

    parent (torch 2.12)  --JSON line-->  worker (torch 2.3.1)  --wav path-->  parent

A persistent process, not one call per utterance -- the model load dominates otherwise.

Protocol: one JSON object per line on stdin, one JSON object per line on stdout.
    in  {"text":..., "instruct":..., "speed":1.0, "prompt_wav":..., "out":...}
    out {"ok":true, "path":..., "dur":...}  |  {"ok":false, "error":...}
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_cosyvoice() -> str:
    """Locate the CosyVoice checkout.

    It is not vendored here -- it is a separate clone with its own submodules and
    pretrained weights. $COSYVOICE_REPO wins; otherwise walk up from this file
    looking for third_party/CosyVoice, which is where the setup notes put it.
    """
    env = os.environ.get("COSYVOICE_REPO")
    if env:
        if not os.path.isdir(env):
            sys.exit(f"COSYVOICE_REPO={env!r} is not a directory")
        return os.path.abspath(env)
    d = HERE
    while True:
        cand = os.path.join(d, "third_party", "CosyVoice")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            sys.exit(
                "CosyVoice checkout not found. Clone it and point COSYVOICE_REPO at it:\n"
                "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice \\\n"
                "      third_party/CosyVoice\n"
                "  export COSYVOICE_REPO=$PWD/third_party/CosyVoice")
        d = parent


REPO = _find_cosyvoice()
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))


def main() -> None:
    model_dir = os.environ.get(
        "COSYVOICE_MODEL",
        os.path.join(REPO, "pretrained_models", "Fun-CosyVoice3-0.5B"))

    import contextlib

    import torch
    import torchaudio

    # CosyVoice (and its deps) log to stdout, which is our protocol channel. Keep stdout
    # clean by routing every library write to stderr; the parent surfaces stderr on error.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    with contextlib.redirect_stdout(sys.stderr):
        from cosyvoice.cli.cosyvoice import CosyVoice3
    
    # fp16 is CUDA-only in this codebase; MPS support inside CosyVoice is partial, so
    # CPU is the reliable choice on Apple silicon. It is a 0.5B model -- workable.
    # CosyVoice3.__init__(model_dir, load_trt, load_vllm, fp16, trt_concurrent)
    # -- no load_jit, unlike CosyVoice/CosyVoice2
        model = CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=False)
        sr = model.sample_rate

        # Memoise the prompt-derived features by reference-clip path.
        #
        # CosyVoice offers add_zero_shot_spk() to avoid re-extracting these, but it
        # is unusable with inference_instruct2: frontend_instruct2 passes
        # instruct_text in the prompt_text slot, and the registered-speaker branch
        # of frontend_zero_shot replaces the whole cached dict, substituting only
        # `text`. So a registered speaker silently discards the per-utterance
        # instruction and feeds the model the reference transcript instead.
        #
        # These three take prompt_wav alone, so caching them by path is equivalent
        # to registration and leaves the instruction channel intact.
        fe = model.frontend
        for _name in ("_extract_speech_token", "_extract_spk_embedding",
                      "_extract_speech_feat"):
            _fn, _cache = getattr(fe, _name), {}
            def _memo(prompt_wav, _fn=_fn, _cache=_cache):
                key = os.path.abspath(prompt_wav) if isinstance(prompt_wav, str) else id(prompt_wav)
                if key not in _cache:
                    _cache[key] = _fn(prompt_wav)
                return _cache[key]
            setattr(fe, _name, _memo)
    print(json.dumps({"ready": True, "sample_rate": sr}), file=real_stdout, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            if job.get("stop"):
                break
            # v3 takes a PATH here (v2 took a 16 kHz tensor as `prompt_speech_16k`);
            # the frontend loads and resamples it itself.
            prompt = job["prompt_wav"]

            # CosyVoice3 asserts on an explicit <|endofprompt|> marker and expects the
            # assistant-style prefix its own examples use; v2 appended this internally.
            instruct = job.get("instruct", "").strip()
            instruct = (f"You are a helpful assistant. {instruct}<|endofprompt|>"
                        if instruct else "You are a helpful assistant.<|endofprompt|>")

            chunks = []
            with contextlib.redirect_stdout(sys.stderr):
              for out in model.inference_instruct2(
                      job["text"], instruct, prompt,
                      stream=False, speed=float(job.get("speed", 1.0))):
                chunks.append(out["tts_speech"])
            wav = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0)
            torchaudio.save(job["out"], wav, sr)
            print(json.dumps({"ok": True, "path": job["out"], "dur": wav.shape[1] / sr}),
                  file=real_stdout, flush=True)
        except Exception as exc:                     # keep the worker alive
            import traceback
            print(json.dumps({"ok": False, "error": f"{exc}",
                              "trace": traceback.format_exc()[-800:]}),
                  file=real_stdout, flush=True)


if __name__ == "__main__":
    main()
