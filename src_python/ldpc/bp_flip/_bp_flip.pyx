#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdio cimport printf
import numpy as np
import sys
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

cdef class BpFlipDecoder(BpDecoderBase):

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, osd_method: int = 0,
                 osd_order: int = 0, flip_iterations: int = 0, pflip_frequency: int = 0, pflip_seed: int = 0):
        
        self.MEMORY_ALLOCATED=False

        ## set up OSD with default values and channel probs from BP
        self.flip_iterations = flip_iterations
        self.flipD = new FlipDecoderCpp(self.pcm[0], self.flip_iterations, pflip_frequency, pflip_seed)
        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.flipD

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
   
        cdef i
        zero_syndrome = True
        out = np.zeros(self.n, dtype=syndrome.dtype)
        
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]:
                zero_syndrome = False
        if zero_syndrome:
            self.bpd.converge = True
            return out
        
        bpd_decoding = self.bpd.decode(self._syndrome)
        if self.bpd.converge:
            for i in range(self.n):
                out[i] = bpd_decoding[i]
        else:
            bpd_syndrome = self.pcm.mulvec(bpd_decoding)
            tmp_syndrome = np.zeros(self.m, dtype=syndrome.dtype)
            for i in range(self.m):
                tmp_syndrome[i] = syndrome[i] ^ bpd_syndrome[i]
            flip_decoding = self.flipD.decode(tmp_syndrome)
            for i in range(self.n):
                out[i] = 0 ###DBG flip_decoding[i] ^ bpd_decoding[i]
        return out

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.flipD.decoding[i]
        return out

    