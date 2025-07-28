#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpDecoderBase
ctypedef np.uint8_t uint8_t

cdef extern from "parbp.hpp" namespace "ldpc::parbp" nogil:

    cdef cppclass ParBpDecoderCpp "ldpc::flip::ParBpDecoder":
        ParBpDecoderCpp(BpSparse& pcm, int max_iter, int pfreq, int seed) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome)
        vector[uint8_t] decoding
        

cdef class ParBpDecoder(BpDecoderBase):
    cdef ParBpDecoderCpp* flipD
    cdef int flip_iterations
