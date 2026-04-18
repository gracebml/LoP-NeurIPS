"""
LoP Algorithms.

Provides training algorithms and helper mechanisms for
Loss of Plasticity research:
    - Backprop (bp): Standard backpropagation
    - ContinualBackprop (cbp): Continual Backprop with generate-and-test
    - SDP: Spectral Diversity Preservation
    - EMAWrapper: Exponential Moving Average for weights
    - GnT / ResGnT: Generate-and-Test variants
"""

from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.algos.sdp import apply_sdp
from lop.algos.ema import EMAWrapper

__all__ = [
    "Backprop",
    "ContinualBackprop",
    "apply_sdp",
    "EMAWrapper",
]
