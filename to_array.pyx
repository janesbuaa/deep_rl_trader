from numpy cimport ndarray as ar
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def to_array(xy):
    cdef int i, j, k, h=len(xy), w=len(xy[0]), p=len(xy[0][0])
    cdef ar[double,ndim=3] new = np.empty((h,w,p))
    for i in xrange(h):
        for j in xrange(w):
            for k in xrange(p):
                new[i,j,k] = xy[i][j][k]
    return new
