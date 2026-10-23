# FastMSS: Fast Multi-Speaker Simulation

FastMSS generates simulated multi-speaker meetings from single-speaker recordings. Turn-taking is sampled from an HMM and rooms are simulated with pyroomacoustics. Besides the mixture, it can write per-speaker anechoic and reverberant references, so the output can be used to train separation and enhancement models as well as speaker-attributed ASR and diarization systems.

## Demo

A simulated meeting with noise and reverberation (click to play on GitHub):

https://github.com/user-attachments/files/26752329/2efd6b4a-8fda-463e-8829-8cf885960f3d.wav

## Speed

<p align="center">
  <img src="docs/scaling_plot_dgx.png" width="500" alt="Scaling benchmark: FastMSS vs MMS-MSG vs NeMo">
</p>

Generation throughput (simulated hours per minute) as a function of the number of parallel workers on a DGX node. Throughput grows almost linearly with the number of workers and is higher than that of the [MMS-MSG](https://github.com/fgnt/mms_msg) and [NeMo](https://github.com/NVIDIA/NeMo) simulators.

## Features

- Turn-taking: an HMM with four transition types (turn hold, turn switch, interruption, backchannel). The transition probabilities can be flat, set by hand or fitted to a corpus (AMI, CallHome, NOTSOFAR-1).
- Overlap control: `boost_overlap_factor` scales the interruption and backchannel rates.
- Rooms: RIRs simulated with pyroomacoustics, with configurable room size and RT60.
- Per-speaker outputs: speaker streams and anechoic references, for separation and enhancement training.
- Noise: recorded noise (for example WHAM or MUSAN) added at a configurable level relative to speech.
- Lhotse: source speech is read from lhotse CutSets with word alignments, and the output manifests are lhotse manifests.
- RTTM and NeMo diarization manifests (optional).
- Multi-channel simulation (optional): table, wall, worn and close-talk arrays, directional talkers and mics, talkers and wearers that move during an utterance, corner noise sources, references aligned to a reference channel. See [Multi-channel simulation](#multi-channel-simulation) and [docs/multichannel.md](docs/multichannel.md).

## Installation

```bash
git clone https://github.com/popcornell/FastMSS.git
cd FastMSS
pip install -e .
```

## Project structure

```
FastMSS/
├── fastmss/                     # Core simulation library
│   ├── simulator.py             # Meeting generation engine
│   ├── hmm_turn_taking.py       # HMM turn-taking model
│   ├── rirsimulator.py          # RIR generation via pyroomacoustics
│   ├── utils.py                 # Audio splitting, crossfading utilities
│   └── spatial/                 # Multi-channel simulation (opt-in)
│       ├── devices.py           # Array/device presets and geometry
│       ├── scene.py             # Stage 3: per-room RIR banks for every device
│       ├── motion.py            # Talker / wearer motion strategies
│       ├── render.py            # Time-varying convolution (RIR interpolation)
│       ├── meeting.py           # Stage 4: multi-channel meeting rendering
│       ├── directivity.py       # Analytic directivity patterns
│       ├── metrics.py           # GCC-PHAT, SRP-PHAT, RT60, DRR, coherence
│       └── config.py            # `spatial` config defaults
├── recipes/
│   ├── sim.py                   # Unified simulation pipeline
│   ├── default.yaml             # Default config
│   ├── multichannel/            # Multi-channel setups (+multichannel=<name>)
│   └── paper/                   # Configs to reproduce paper experiments
│       ├── ts_asr/              # TS-ASR experiment configs (Table 1 & 2)
│       └── diarization/         # Diarization experiment configs
├── preprocessing/
│   ├── download_wham.sh         # Download WHAM noise dataset
│   ├── download_otospeech.sh    # Download + prepare otoSpeech dataset
│   └── resample_folder.py       # General audio resampling utility
├── tests/                       # pytest suite (legacy regression, spatial checks)
├── setup.py
└── requirements.txt
```

## Quick start

### 1. Prepare source data

FastMSS takes single-speaker audio in [lhotse](https://github.com/lhotse-speech/lhotse) format as input. Any speech corpus can be used. Two options:

```bash
# Option A: LibriSpeech (via lhotse)
lhotse download librispeech --full /path/to/librispeech
lhotse prepare librispeech /path/to/librispeech /path/to/manifests

# Option B: otoSpeech (from HuggingFace)
bash preprocessing/download_otospeech.sh /path/to/manifests /path/to/otospeech

# Optional: download WHAM noise for augmentation
bash preprocessing/download_wham.sh /path/to/wham
```

### 2. Generate meetings

All paths are set with Hydra overrides on the command line:

```bash
# Basic: 1000 meetings, 2-4 speakers, 60s each, flat turn-taking priors
python recipes/sim.py \
    output_dir=/path/to/output \
    manifest_dir=/path/to/manifests \
    n_meetings=1000 \
    duration=60 \
    n_jobs=16

# With noise + reverberation
python recipes/sim.py \
    output_dir=/path/to/output \
    manifest_dir=/path/to/manifests \
    noise_folders=[/path/to/wham] \
    add_noise=true \
    reverberate=true \
    n_jobs=16

# Save per-speaker anechoic references (for separation/enhancement)
python recipes/sim.py \
    output_dir=/path/to/output \
    manifest_dir=/path/to/manifests \
    save_spk=true \
    save_anechoic=true \
    reverberate=true \
    n_jobs=16

# Generate RTTM + NeMo manifests (for diarization)
python recipes/sim.py \
    output_dir=/path/to/output \
    manifest_dir=/path/to/manifests \
    save_rttm=true \
    save_nemo_manifest=true \
    n_jobs=16
```

### Main parameters

General:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Where to write simulated audio and manifests | required |
| `manifest_dir` | Directory with lhotse source manifests | required |
| `n_meetings` | Number of meetings to generate | `1000` |
| `duration` | Target meeting duration in seconds | `60` |
| `min_max_spk` | Speaker count range `[min, max]` | `[2, 4]` |
| `n_jobs` | Parallel workers | `8` |

Augmentation (each option is independent):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reverberate` | Apply room impulse responses to the mix | `false` |
| `reverb_prob` | Probability of applying RIR per meeting (if `reverberate=true`) | `0.5` |
| `rt60` | RT60 range in seconds | `[0.1, 0.7]` |
| `room_sz` | Room dimensions range in meters | `[5, 10]` |
| `n_rirs` | Number of RIRs to pre-simulate | `5000` |
| `add_noise` | Add background noise to the mix | `false` |
| `noise_folders` | Noise audio directories (list) | `null` |
| `noise_rel_gain` | Noise gain relative to speech (dB range) | `[-20, -3]` |
| `noise_probability_global` | Probability of adding noise per meeting | `0.5` |
| `speed_perturb` | Apply speed perturbation to source utterances | `false` |
| `speed_perturb_range` | Speed perturbation factor range | `[0.95, 1.05]` |
| `use_fir` | Apply FIR filtering | `false` |

Output:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `save_spk` | Save per-speaker audio streams | `false` |
| `save_anechoic` | Save anechoic (dry) per-speaker references | `false` |
| `save_rttm` | Generate RTTM files (Stage 5) | `false` |
| `save_nemo_manifest` | Generate NeMo-style diarization manifest | `false` |

Turn-taking:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hmm_params.p_ind` | Transition probabilities `[hold, switch, interruption, backchannel]` | `[0.25, 0.25, 0.25, 0.25]` |
| `hmm_fit_transitions_to` | Fit HMM to a target corpus (path to lhotse manifest) | `null` |
| `boost_overlap_factor` | Scale interruption/backchannel rates | `null` |
| `use_markov` | Use Markov chain for state transitions | `false` |
| `stage` | Start from this pipeline stage | `0` |

## Multi-channel simulation

Configs without a `spatial` block, or with `spatial.enabled: false`, use the
single-channel path described above, and its output is unchanged. With
`spatial.enabled: true`, stages 3 and 4 simulate multi-channel devices. The
recipes in `recipes/multichannel/` set up common devices:

```bash
# 8-channel AMI-like table array; other recipes: notsofar1, kinect, chime6, easycom, closetalk
python recipes/sim.py +multichannel=ami \
    output_dir=./output/mc_ami manifest_dir=<MANIFEST_DIR> \
    add_noise=true 'noise_folders=[<NOISE_DIR>]' save_spk=true save_anechoic=true

# any spatial key can be overridden, for example
python recipes/sim.py +multichannel=easycom ... spatial.motion.move_prob=0.5 spatial.reference.dump=all
```

| Preset | Channels | Placement |
|--------|----------|-----------|
| `single` | 1 | free |
| `circular` | `n_mics` (+1 center) | table |
| `ami` | 8 | table |
| `notsofar1` | 7 | table |
| `kinect` | 4 | wall |
| `binaural_worn` | 2 | head |
| `glasses` | 6 | head |
| `closetalk` | 1 per talker | mouth |

[docs/multichannel.md](docs/multichannel.md) describes the devices, the RIR
banks, motion, the time-varying convolution, reference alignment and the
noise model, with figures and the results of the checks. The tests run with
`python -m pytest tests/`.

## Turn-taking model

FastMSS uses a 4-state HMM to model conversational dynamics. At each turn boundary, the model samples one of:

| Transition | Description | Effect |
|------------|-------------|--------|
| Turn hold | Same speaker continues | No overlap, pause |
| Turn switch | New speaker takes the floor | No overlap, speaker change |
| Interruption | New speaker starts before current finishes | Partial overlap |
| Backchannel | Brief interjection while speaker continues | Short overlap |

Transition probabilities can be set to uniform (`flat`), fitted to a target corpus via `hmm_fit_transitions_to`, or manually specified via `hmm_params.p_ind`. The `boost_overlap_factor` parameter scales the interruption and backchannel probabilities to increase or decrease overall overlap.

## Reproducing the paper experiments

The `recipes/paper/` directory contains the exact configs used in the Interspeech 2026 paper. To run them, override the config name:

```bash
# TS-ASR Table 1: flat turn-taking priors
python recipes/sim.py \
    --config-name paper/ts_asr/table1/flat \
    output_dir=/path/to/output \
    manifest_dir=/path/to/manifests

# Diarization: 2-speaker clean with overlap boosting
python recipes/sim.py \
    --config-name paper/diarization/clean/librispeech_2spk \
    output_dir=/path/to/output \
    manifest_dir=/path/to/manifests
```

## Citation

If you use FastMSS in your research, please cite:

```bibtex
@inproceedings{polok2026mindthegap,
    author = {Polok, Alexander and Medennikov, Ivan and Watanabe, Shinji and {\v{C}}ernock{\'y}, Jan and Burget, Luk{\'a}{\v{s}} and Cornell, Samuele},
    title = {Mind the Gap: Impact of Synthetic Conversational Data on Multi-Talker {ASR} and Speaker Diarization},
    booktitle = {Proc. Interspeech},
    year = {2026}
}
```

## License

FastMSS is released under the Apache License, Version 2.0 (see `LICENSE` and `NOTICE`).
