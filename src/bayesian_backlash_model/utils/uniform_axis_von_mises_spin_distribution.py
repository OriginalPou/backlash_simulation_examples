# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 05/06/2024

Define a uniform-axis random spin distribution for rotations in SO(3)

"""
#%%
from particles.distributions import ProbDist
import numpy as np
from scipy.stats import vonmises
from scipy.special import i0
from scipy.spatial.transform import Rotation

class UARS_von_Mises(ProbDist):
    """
    uniform-axis random spin distribution for rotations in SO(3)

    The distribution is defined by 
        - an axis that is uniformly distributed on the unit
        - an angle that independently distributed according to the circular-von Mises distribution
    
    Parameters
    ----------
    mode : ndarray
        Rotation matrix in SO(3) representing the _mean_ rotation of the distribution
    kappa : float
        concentration parameter for the von Mises distribution
    
    References
    ----------
    Modeling and Inference for Measured Crystal Orientations and a Tractable Class of Symmetric
    Distributions for Rotations in Three Dimensions, 
    
    Author : Bingham et al.
    
    URL : https://www.jstor.org/stable/40592347
    """


    def __init__(self, kappa=1.0, mode = np.eye(3)):
        self.kappa = kappa
        self.mode = mode

    @property
    def dim(self):
        return 3
    
    def pdf(self,x):
        '''
        Parameters
        ----------
            x : ndarray (Dx3x3)
                Rotation matrix in SO(3)
        '''
        dR = Rotation.from_matrix(np.transpose(x.T @ self.mode,(0,2,1)))
        return (vonmises(kappa = self.kappa).pdf(np.arccos(1/2 * (np.trace(dR.as_matrix(), axis1=-2, axis2=-1) - 1))))
    
    def logpdf(self, x):
        '''
        Parameters
        ----------
            x : ndarray (3x3)
                Rotation matrix in SO(3)
        '''
        return np.log(self.pdf(x))
    
    def rvs(self, size = 1):
        '''
        generates *size* variates from distribution
        
        Returns
        -------
            samples : ndarray
                (size, 3, 3) array of rotation matrices if size > 1, (3,3) rotation matrix if size = 1
        '''
        # sample rotation axis uniformly on the unit sphere by using spherical coordinates
        # axis = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
        theta = np.random.uniform(low=0, high= np.pi, size= size)
        phi = np.random.uniform(low=0, high= 2* np.pi, size= size)
        axis = np.array([np.sin(theta)*np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        # sample rotation angle according to the von Mises distribution
        angle = vonmises(loc = 0, kappa = self.kappa).rvs(size = size).reshape(size, 1)
        # use axis and rotation angle to obtain rotation matrices
        samples = self.mode @ Rotation.from_rotvec(angle * axis.T).as_matrix()
        return(samples.reshape((3,3)) if size == 1 else samples)


# %%
