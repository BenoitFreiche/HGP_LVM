# cython: infer_types=True
import numpy as np
cimport cython

DTYPE = np.intc

@cython.boundscheck(False)
@cython.wraparound(False)
cdef det3(double[:] a0, double[:] a1, double[:] a2):
    return (a0[0] * (a1[1] * a2[2] - a2[1] * a1[2])
           -a1[0] * (a0[1] * a2[2] - a2[1] * a0[2])
           +a2[0] * (a0[1] * a1[2] - a1[1] * a0[2]))

@cython.boundscheck(False)
@cython.wraparound(False)
def volumeSubpart(double[:, :] points, int[:, :]triangles):
    cdef double acumVol = 0
    cdef int i 
    #cdef float[:] mean = np.mean(points, axis = 0)
    for i in range(triangles.shape[0]):
        acumVol += det3(points[triangles[i][0]], points[triangles[i][1]], points[triangles[i][2]] )/6
    return np.abs(acumVol)

