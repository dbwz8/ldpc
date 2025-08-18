from libc.stdio import printf
import numpy as np
import sys
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

class BpFlipDecoder(BpDecoderBase): ...

    def __del__(self): ...

    def decode(self, syndrome: np.ndarray) -> np.ndarray: ...
    def decode(self, syndrome: np.ndarray) -> np.ndarray: ...

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
