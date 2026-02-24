from pathlib import Path
import json

import numpy as np
import pyroomacoustics
import soundfile as sf
import math

from typing import Iterable


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

    @staticmethod
    def create_circular_mic_array(center, diameter, num_mics):
        """
        Creates a circular microphone array around a center point.

        Parameters:
        center (list): 3D coordinates [x, y, z] of the center point
        diameter (float): Diameter of the circular array
        num_mics (int): Number of microphones in the array

        Returns:
        list: List of 3D coordinates, each representing a microphone position
        """
        if isinstance(num_mics, Iterable):
            num_mics = np.random.randint(num_mics[0], num_mics[-1] + 1)

        if isinstance(diameter, Iterable):
            diameter = np.random.uniform(diameter[0], diameter[-1] + 1e-8)

        if num_mics <= 0:
            return []

        radius = diameter / 2.0
        mic_positions = []

        # Calculate angular separation between microphones
        angle_step = 2 * math.pi / num_mics

        for i in range(num_mics):
            # Calculate angle for this microphone
            angle = i * angle_step

            # Calculate position relative to center
            # Assuming the array lies in the XY plane (constant Z)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]  # Keep same Z coordinate as center

            mic_positions.append([x, y, z])

        return mic_positions

    def sample_linear_array(self, ):
        raise NotImplementedError

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


        if self.cfg.mic_type == "single":
            mic_locs = np.array([mic_x, mic_y, mic_z])
            anechoic.add_microphone(mic_locs)
            room.add_microphone(mic_locs)
        elif self.cfg.mic_type == "circular":
            mic_locs = np.array([mic_x, mic_y, mic_z])
            circ_arr_coords = self.create_circular_mic_array([mic_x, mic_y, mic_z], self.cfg.circular.diameter, self.cfg.circular.n_mics)
            for coord in circ_arr_coords:
                room.add_microphone(np.array(coord))
                anechoic.add_microphone(np.array(coord))
        else:
            raise NotImplementedError



        # Store positions for later saving
        src_positions = []
        for src in range(self.cfg.n_pos_rirs):
            tmp = self.sample_src_pos(l, w, mic_locs)
            src_positions.append(tmp.tolist())  # Convert to list for JSON serialization
            room.add_source(tmp)
            anechoic.add_source(tmp)

        room.compute_rir()
        anechoic.compute_rir()
        output_dir = Path(self.cfg.output_dir, "rirs", f"{meeting_id}").absolute()
        output_dir.mkdir(exist_ok=True, parents=True)
        #

        output_rirs = []
        position_metadata = []
        for src in range(self.cfg.n_pos_rirs):
            maxlen = max([len(room.rir[x][src]) for x in range(len(room.rir))])
            # pad and then save jointly.
            tmp = []
            for mic_inx in range(len(room.rir)):
                c_rir = room.rir[mic_inx][src]
                if c_rir.shape[0] < maxlen:
                    c_rir = np.pad(c_rir, (0, maxlen - c_rir.shape[0]))
                tmp.append(c_rir)
            c_rir = np.stack(tmp, -1)
            # need to handle multichannel here, 0 is the first channel.
            flac_path = str(output_dir / Path(meeting_id + f"_{src}.flac"))
            output_rirs.append(flac_path)
            sf.write(flac_path, c_rir, samplerate=self.cfg.samplerate)

            # Store position metadata
            position_metadata.append({
                "rir_file": flac_path,
                "position": src_positions[src]
            })

        for src in range(self.cfg.n_pos_rirs):
            c_rir = anechoic.rir[0][src]
            # for anechoic just use the first channel always !
            flac_path = str(output_dir / Path(meeting_id + f"_{src}-anechoic.flac"))
            sf.write(flac_path, c_rir, samplerate=self.cfg.samplerate)

        # Save position metadata to JSON
        metadata_path = output_dir / f"{meeting_id}_positions.json"
        with open(metadata_path, 'w') as f:
            json.dump({"positions": position_metadata}, f, indent=2)

        return output_rirs
