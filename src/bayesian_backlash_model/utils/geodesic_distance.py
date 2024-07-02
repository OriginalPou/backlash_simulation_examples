import numpy as np
from copy import deepcopy

from ukfm import SO3


def geodesic_distance(R1 : np.ndarray, R2 : np.ndarray) :
    '''
    Computes the geodesic distance between two rotation matrices

    Parameters
    ----------
    R1 : ndarray
        rotation matrix
    R2 : ndarray
        rotation matrix

    Returns
    -------
    the geodesic distance between R1 and R2 : ndarray
    '''
    r1,p1,y1 = SO3.to_rpy(R1)
    r2,p2,y2 = SO3.to_rpy(R2)

    R1 = deepcopy(SO3.from_rpy(r1, p1, y1))
    R2 = deepcopy(SO3.from_rpy(r2, p2, y2))

    dist = (np.trace(np.transpose(R1) @ R2)-1)/2
    # make sure dist \in [-1, 1] before passing it to arccos
    # dist = 1 if dist > 1 else -1 if dist <-1 else dist 
    return(np.abs(np.arccos(dist)))
