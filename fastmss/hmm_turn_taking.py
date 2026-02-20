from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from lhotse import SupervisionSet


class TransitionType(Enum):
    """Four types of utterance transitions as defined in the paper"""

    TURN_HOLD = "TH"  # Same speaker with pause
    TURN_SWITCH = "TS"  # Different speaker with gap
    INTERRUPTION = "IR"  # Different speaker with overlap
    BACKCHANNEL = "BC"  # Different speaker fully overlapped


@dataclass
class TransitionParams:
    """Parameters for each transition type"""

    beta_th: float = 0.57  # Expected pause duration for turn-hold
    beta_ts: float = 0.40  # Expected gap duration for turn-switch
    beta_ir: float = 0.24  # Expected overlap ratio for interruption
    beta_bc: float = 0.47  # Expected overlap ratio for backchannel # not used now

    # Probability distributions
    p_ind: List[float] = None  # [p_TH, p_TS, p_IR, p_BC] for random selection
    p_markov: np.ndarray = None  # 4x4 transition matrix for Markov selection

    def __post_init__(self):
        if self.p_ind is None:
            # Default probabilities from CALLHOME1 (Table 1 in paper)
            self.p_ind = [0.15, 0.21, 0.44, 0.20]

        if self.p_markov is None:
            # Default Markov transition matrix from CALLHOME1
            self.p_markov = np.array(
                [
                    [0.26, 0.11, 0.09, 0.31],  # TH -> [TH, TS, IR, BC]
                    [0.23, 0.38, 0.29, 0.29],  # TS -> [TH, TS, IR, BC]
                    [0.27, 0.31, 0.33, 0.31],  # IR -> [TH, TS, IR, BC]
                    [0.24, 0.20, 0.29, 0.09],  # BC -> [TH, TS, IR, BC]
                ]
            ).T  # need to take transpose here

    def fit(self, supervisions: SupervisionSet, bc_duration_threshold: float = 1.0):
        """
        Fits parameters on a Lhotse SupervisionSet.
        Includes strict normalization to ensure probabilities sum to exactly 1.0.
        """
        if not supervisions:
            return

        # 1. Group by recording
        recs = defaultdict(list)
        for s in supervisions:
            recs[s.recording_id].append(s)

        # Accumulators
        # [TH, TS, IR, BC]
        value_bins = {self.IDX_TH: [], self.IDX_TS: [], self.IDX_IR: [], self.IDX_BC: []}
        type_counts = np.zeros(4)
        trans_counts = np.zeros((4, 4))  # Rows=Prev, Cols=Next

        for rid, segs in recs.items():
            # Sort chronological
            segs.sort(key=lambda x: x.start)

            prev_type_idx = None

            for i in range(1, len(segs)):
                prev_seg = segs[i - 1]
                curr_seg = segs[i]

                # Calculate timing
                gap = curr_seg.start - prev_seg.end

                if curr_seg.speaker == prev_seg.speaker:
                    # Turn Hold (TH)
                    current_type_idx = self.IDX_TH
                    metric_value = max(0.0, gap)
                else:
                    # Different speaker
                    if gap >= 0:
                        # Turn Switch (TS)
                        current_type_idx = self.IDX_TS
                        metric_value = gap
                    else:
                        # Overlap: Interruption (IR) or Backchannel (BC)
                        overlap_duration = abs(gap)

                        # Heuristic: Short utterances (<= 1s) are Backchannels
                        if curr_seg.duration <= bc_duration_threshold:
                            current_type_idx = self.IDX_BC
                        else:
                            current_type_idx = self.IDX_IR

                        # Overlap Ratio
                        if curr_seg.duration > 0:
                            metric_value = overlap_duration / curr_seg.duration
                        else:
                            metric_value = 0.0
                        metric_value = min(1.0, metric_value)

                if current_type_idx is not None:
                    value_bins[current_type_idx].append(metric_value)
                    type_counts[current_type_idx] += 1

                    if prev_type_idx is not None:
                        trans_counts[prev_type_idx, current_type_idx] += 1

                    prev_type_idx = current_type_idx

        # 1. Update Betas (Means)
        # Only update if observed data points to avoid zero-division or NaN
        if len(value_bins[self.IDX_TH]) > 0: self.beta_th = float(np.mean(value_bins[self.IDX_TH]))
        if len(value_bins[self.IDX_TS]) > 0: self.beta_ts = float(np.mean(value_bins[self.IDX_TS]))
        if len(value_bins[self.IDX_IR]) > 0: self.beta_ir = float(np.mean(value_bins[self.IDX_IR]))
        if len(value_bins[self.IDX_BC]) > 0: self.beta_bc = float(np.mean(value_bins[self.IDX_BC]))

        # 2. Update Independent Probabilities (p_ind)
        # Strict normalization: Last element = 1.0 - sum(others)
        total_transitions = np.sum(type_counts)
        if total_transitions > 0:
            probs = type_counts / total_transitions
            # Fix floating point drift
            probs[-1] = 1.0 - np.sum(probs[:-1])
            # Ensure no negative small numbers (e.g. -1e-10) due to precision
            probs = np.maximum(probs, 0.0)
            # Re-normalize one last time just in case clipping happened, though rarely needed after remainder method
            if probs.sum() > 0:
                probs /= probs.sum()
            self.p_ind = probs.tolist()

        # 3. Update Markov Matrix (p_markov)
        # Strict normalization per row
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        # Handle rows with 0 transitions (avoid div by zero)
        row_sums[row_sums == 0] = 1.0

        norm_trans = trans_counts / row_sums

        # Apply strict sum=1.0 check for every row
        for r in range(4):
            if np.sum(trans_counts[r]) > 0:
                norm_trans[r, -1] = 1.0 - np.sum(norm_trans[r, :-1])

        # Clip negatives and re-normalize for safety
        norm_trans = np.maximum(norm_trans, 0.0)
        norm_trans /= norm_trans.sum(axis=1, keepdims=True)

        # Transpose to match the shape expected in __post_init__
        self.p_markov = norm_trans.T
