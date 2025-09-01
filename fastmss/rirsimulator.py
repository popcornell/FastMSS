from pathlib import Path

import numpy as np
import pyroomacoustics
import soundfile as sf


class RIRSimulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample_src_pos(self, l, w, mic_pos):

        def distance(current_points, ref_point):
            # current_points = np.array(current_points)
            # ref_point = np.array(ref_point)
            return np.sqrt(np.sum((current_points - ref_point) ** 2, axis=0))

        delta_dist = self.cfg.delta_dist
        while True:
            src_x = np.random.uniform(delta_dist, l - delta_dist)
            src_y = np.random.uniform(delta_dist, w - delta_dist)
            src_z = np.random.uniform(0.5, 2)
            c_src_pos = np.array([src_x, src_y, src_z])
            if distance(c_src_pos, mic_pos) > 1:
                return c_src_pos

    def gen_rirs(self, meeting_id):
        # sample some amount of RIRs for each meeting and iterate over these.
        # so that speaker can change position
        while True:
            l = np.random.uniform(*self.cfg.room_sz)
            w = np.random.uniform(l / 2, self.cfg.room_sz[-1])
            z = np.random.uniform(*self.cfg.room_ceiling)
            room_dim = [l, w, z]
            rt60 = np.random.uniform(*self.cfg.rt60)  # sample random RT60
            try:
                e_absorption, max_order = pyroomacoustics.inverse_sabine(rt60, room_dim)
                # retry till we can generate a valid room. This happens when rt60 is too low for a huge room.
            except ValueError:
                continue

            if self.cfg.save_anechoic:
                if self.cfg.use_rand_ism is not False or self.cfg.rand_disp != 0.0:
                    raise ValueError
                    (
                        "If you want to save "
                        "the anechoic signal, pyroomacoustics "
                        "use_rand_ism should be False and rand_disp should be 0.0."
                    )

            room = pyroomacoustics.ShoeBox(
                room_dim,
                fs=self.cfg.samplerate,
                max_order=max_order,
                materials=pyroomacoustics.Material(e_absorption),
                use_rand_ism=self.cfg.use_rand_ism,
                max_rand_disp=self.cfg.rand_disp,
            )
            break

        # create also direct path room here
        pass
        anechoic = pyroomacoustics.ShoeBox(
            room_dim,
            fs=self.cfg.samplerate,
            max_order=0,  # only direct sound
            materials=pyroomacoustics.Material(e_absorption),
            use_rand_ism=self.cfg.use_rand_ism,
            max_rand_disp=self.cfg.rand_disp,
        )
        delta_dist = self.cfg.delta_dist
        mic_x = np.random.uniform(delta_dist, l - delta_dist)
        mic_y = np.random.uniform(delta_dist, w - delta_dist)
        mic_z = np.random.uniform(0.5, z - delta_dist)
        # add microphone in the room randomly

        mic_locs = np.array([mic_x, mic_y, mic_z])
        room.add_microphone(mic_locs)
        anechoic.add_microphone(mic_locs)

        for src in range(self.cfg.n_pos_rirs):
            tmp = self.sample_src_pos(l, w, mic_locs)
            room.add_source(tmp)
            anechoic.add_source(tmp)

        room.compute_rir()
        anechoic.compute_rir()
        output_dir = Path(self.cfg.output_dir, "rirs", f"{meeting_id}").absolute()
        output_dir.mkdir(exist_ok=True, parents=True)
        # maxlen = max([len(room.rir[0][x]) for x in range(n_positions)])

        output_rirs = []
        for src in range(self.cfg.n_pos_rirs):
            c_rir = room.rir[0][src]
            # need to handle multichannel here, 0 is the first channel.
            flac_path = str(output_dir / Path(meeting_id + f"_{src}.flac"))
            output_rirs.append(flac_path)
            sf.write(flac_path, c_rir, samplerate=self.cfg.samplerate)

        for src in range(self.cfg.n_pos_rirs):
            c_rir = anechoic.rir[0][src]
            # need to handle multichannel here, 0 is the first channel.
            flac_path = str(output_dir / Path(meeting_id + f"_{src}-anechoic.flac"))
            sf.write(flac_path, c_rir, samplerate=self.cfg.samplerate)

        return output_rirs
