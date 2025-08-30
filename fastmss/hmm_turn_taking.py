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

    beta_th: float = 0.77  # Expected pause duration for turn-hold
    beta_ts: float = 0.60  # Expected gap duration for turn-switch
    beta_ir: float = 0.44  # Expected overlap ratio for interruption
    beta_bc: float = 0.67  # Expected overlap ratio for backchannel # not used now

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

    def fit(self, supervisions: Optional[SupervisionSet] = None):
        raise NotImplementedError  # TODO this would be cool, fit on supervisionset
        pass
