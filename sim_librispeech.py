import glob
import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import hydra
import lhotse
import soundfile as sf
from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.manipulation import combine as combine_manifests
from lhotse.parallel import parallel_map
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastmss.rirsimulator import RIRSimulator
from fastmss.simulator import ConversationalMeetingSimulator
from fastmss.utils import split_monocuts_batch

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discard(x, duration):
    info = sf.SoundFile(x)
    # allow for a little longer since the meetings can be a bit longer when there
    # are many speakers...
    if (len(info) / info.samplerate) > (duration):
        return x
    else:
        return None


@hydra.main(config_path="config", config_name="librispeech")
def main(cfg: DictConfig) -> None:

    if cfg.stage <= 0:
        lhotse_manifest_dir = os.path.join(cfg.output_dir, "manifests")
        Path(lhotse_manifest_dir).mkdir(parents=True, exist_ok=True)
        from lhotse.recipes.librispeech import prepare_librispeech

        prepare_librispeech(
            corpus_dir=cfg.librispeech_dir,
            alignments_dir=cfg.librispeech_align,
            output_dir=os.path.join(cfg.output_dir, "manifests"),
            dataset_parts=cfg.dset_splits,
            num_jobs=cfg.n_jobs,
        )

        all_cuts = []
        for split in cfg.dset_splits:
            c_rec = lhotse.load_manifest(os.path.join(cfg.output_dir, "manifests",
                f"{cfg.manifest_prefix}_recordings_{split}.jsonl.gz"))
            c_sup = lhotse.load_manifest(
                os.path.join(cfg.output_dir, "manifests", f"{cfg.manifest_prefix}_supervisions_{split}.jsonl.gz"
            ))
            c_cut = CutSet.from_manifests(recordings=c_rec, supervisions=c_sup)
            all_cuts.append(c_cut)
        all_cuts = combine_manifests(all_cuts)
        logger.info("Saving source CutSet to disk")

        all_cuts.to_file(
            os.path.join(cfg.output_dir, "manifests", "all_cuts_orig.jsonl.gz")
        )

    if cfg.stage <= 1:
        try:
            all_cuts
        except NameError:
            logger.info("Loading source CutSet from disk")
            all_cuts = lhotse.load_manifest(
                os.path.join(cfg.output_dir, "manifests", "all_cuts_orig.jsonl.gz")
            )
        logger.info(f"Before splitting with forced alignment: {len(all_cuts)} cuts.")
        all_cuts = split_monocuts_batch(
            all_cuts, cfg.split_fa_factor, num_jobs=cfg.n_jobs)
        logger.info(f"After splitting with forced alignment: {len(all_cuts)} cuts.")
        logger.info(f"Saving to disk splitted cuts.")
        if len(all_cuts) == 0:
            raise RuntimeError("No cuts left, did you specify the correct path for the forced alignment ?. Exiting.")
        all_cuts.to_file(os.path.join(cfg.output_dir, "manifests", "all_cuts.jsonl.gz"))

    if cfg.stage <= 2:
        if cfg.noise_folders is not None:
            logger.info("Parsing background noise files.")
            noise_files = []
            for c_folder in cfg.noise_folders:
                for c_ext in [".wav", ".flac", ".mp3"]:
                    tmp = glob.glob(
                        os.path.join(c_folder, "**/*" + c_ext), recursive=True
                    )
                    noise_files.extend(tmp)

            # Apply regex filter if specified
            if hasattr(cfg, 'noise_filename_pattern') and cfg.noise_filename_pattern is not None:
                pattern = re.compile(cfg.noise_filename_pattern)
                original_count = len(noise_files)
                noise_files = [f for f in noise_files if pattern.search(os.path.basename(f))]
                filtered_count = original_count - len(noise_files)
                logger.info(
                    f"Applied filename pattern '{cfg.noise_filename_pattern}': "
                    f"kept {len(noise_files)} files, filtered out {filtered_count} files."
                )

            worker = partial(discard, duration=cfg.filter_noise_len)
            # filter noise that are too short
            filtered = []
            for n in tqdm(
                parallel_map(worker, noise_files, num_jobs=cfg.n_jobs),
                total=len(noise_files),
                desc="Parsing noise files.",
            ):
                filtered.append(n)

            filtered = [x for x in filtered if x is not None]
            diff = len(noise_files) - len(filtered)
            logger.info(
                f"Discarded {diff} noise files as they were shorter than {cfg.filter_noise_len}. Now {len(filtered)}, before {len(noise_files)}."
            )
            noise_files = filtered

            assert len(noise_files) > 0, "No noise files found, wrong path ?"
            out_file = os.path.join(cfg.output_dir, "manifests", "noise_files.txt")
            with open(out_file, "w") as f:
                f.writelines([str(x) + "\n" for x in noise_files])
            # e.g. WHAM or SINS
            logger.info(f"Noise files paths saved in {out_file}")
            # TODO impulsive noises ?
        else:
            noise_files = None

    if cfg.stage <= 3:
        logger.info("Simulating RIRs using Pyroomacoustics")

        simulator = RIRSimulator(cfg)

        worker = partial(simulator.gen_rirs)

        meeting_ids = iter([f"room_{x}" for x in range(cfg.n_rirs)])
        all_rooms = []
        for c_rirs in tqdm(
            parallel_map(worker, meeting_ids, num_jobs=cfg.n_jobs),
            total=cfg.n_rirs,
            desc="Simulating room impulse responses (RIRs)",
        ):
            all_rooms.append(c_rirs)

        # JSON file here with a list of lists
        # each element will be a single RIR file (can be multichannel)
        out_file = os.path.join(cfg.output_dir, "manifests", "all_rooms.json")
        with open(out_file, "w") as f:
            json.dump(all_rooms, f, indent=4)

    if cfg.stage <= 4:
        try:
            all_cuts
        except NameError:
            logger.info("Loading source CutSet")
            all_cuts = lhotse.load_manifest(
                os.path.join(cfg.output_dir, "manifests", "all_cuts.jsonl.gz")
            )
            logger.info("Source CutSet loaded")

        if cfg.noise_folders is not None:
            # load noise files
            out_file = os.path.join(cfg.output_dir, "manifests", "noise_files.txt")
            with open(out_file, "r") as f:
                noise_files = f.readlines()
            noise_files = [x.strip("\n") for x in noise_files]
        else:
            noise_files = None

        if cfg.reverberate == True:
            # load rirs JSON
            out_file = os.path.join(cfg.output_dir, "manifests", "all_rooms.json")
            with open(out_file, "r") as f:
                rirs_files = json.load(f)

            # Load position metadata for each room
            rirs = []
            for room_rirs in rirs_files:
                # Get room directory from first RIR file
                room_dir = Path(room_rirs[0]).parent
                room_id = Path(room_rirs[0]).stem.rsplit('_', 1)[0]  # Extract room_id from filename

                # Load position metadata
                metadata_file = room_dir / f"{room_id}_positions.json"
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Create mapping from RIR file to position
                rir_to_pos = {item['rir_file']: item['position'] for item in metadata['positions']}

                # Build room RIRs with positions
                room_rirs_with_pos = []
                for rir_file in room_rirs:
                    room_rirs_with_pos.append({
                        'file': rir_file,
                        'pos': rir_to_pos[rir_file]
                    })
                rirs.append(room_rirs_with_pos)
        else:
            rirs = None

        simulator = ConversationalMeetingSimulator(
            cfg,
            Path(cfg.output_dir).absolute() / Path("audio"),
            all_cuts,
            rirs=rirs,
            noise_files=noise_files,
        )

        output_dir = Path(cfg.output_dir).absolute() / Path("manifests")
        output_dir.mkdir(exist_ok=True, parents=True)

        uuids = iter([f"simulation_{x}" for x in range(cfg.n_meetings)])
        work = partial(simulator.gen_audio)
        # for i in range(10):
        #    simulator.gen_audio(i)
        # raise BufferError
        recordings = []
        supervisions = []
        for c_rec, c_sup in tqdm(
            parallel_map(work, uuids, num_jobs=cfg.n_jobs, threads=True),
            total=cfg.n_meetings,
            desc="Simulating meetings",
        ):
            recordings.append(c_rec)
            supervisions.extend(c_sup)

        supervisions = SupervisionSet(supervisions)
        recordings = RecordingSet(recordings)
        lhotse.validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )
        logger.info("Saving simulated manifests to disk")
        supervisions.to_file(
            output_dir / f"synth-{cfg.manifest_prefix}-train-supervisions.jsonl.gz"
        )
        recordings.to_file(
            output_dir / f"synth-{cfg.manifest_prefix}-train-recordings.jsonl.gz"
        )

        if cfg.save_cutset:
            cutset = lhotse.CutSet.from_manifests(
                recordings=recordings, supervisions=supervisions
            )
            # save also cutset here.
            cutset.to_file(
                output_dir / f"synth-{cfg.manifest_prefix}-train-cuts.jsonl.gz"
            )


if __name__ == "__main__":
    main()
