# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 27/05/2024
"""
import numpy as np
from ukfm import SO3
from copy import deepcopy

from bayesian_backlash_model.models.backlash.sinkhole_backlash_model import BacklashSinkhole

class EndoscopeMotionModel():
    '''
    Motion model of a two bending plane endoscope

    The motion model expresses
        - the attitude of the tip of the endoscope over time
        - the direction of motion of the tip of the endoscope over time
    
    Attitude
    --------
        Computing the attitude is based 
            - the angular Jacobian of the robot tip w.r.t each of the actuators
            - the backlash model of each one of the actuators
        
        Multiply the angular Jacobian and the output of the backlash model to
        obtain a tangent vector to the SO(3) manifold

        Use the Exponential map of SO(3) to obtain the attitude of the robot tip
        at the current timestep w.r.t the previous timestep
    
    .. math::
    R_n^w = R_{n-1}^w @ R_n^{n-1}
    R_n^{n-1} = Exp_{SO(3)} (
    
                J_{\Omega(q_1)} * Bl(m_{q_1}, w_{q_1})(q_1)  + 

                J_{\Omega(q_2)} * Bl(m_{q_2}, w_{q_2})(q_2) )

    J_{\Omega(q)} = [J_{\Omega_x(q)}, J_{\Omega_y(q)}, J_{\Omega_z(q)}]
    '''
    def __init__(self, params : np.ndarray):
        '''
        Parameters
        ----------
            params : ndarray
                m_{q_1}, w_{q_1},                                           # Backlash model params for act 1
                      
                m_{q_2}, w_{q_2},                                           # Backlash model params for act 2
                      
                J_{\Omega_x(q1)}, J_{\Omega_y(q1)}, J_{\Omega_z(q1)},       # Angular Jacobian for act 1
                      
                J_{\Omega_x(q2)}, J_{\Omega_y(q2)}, J_{\Omega_z(q2)}        # Angular Jacobian for act 2
        '''
        self.bl_q1_params = {'m': params[0], 'w': params[1]}
        self.bl_q2_params = {'m': params[2], 'w': params[3]}
        
        self.ang_jac_q1 = params[4:7]
        self.ang_jac_q2 = params[7:10]

        self.BL_q1 = BacklashSinkhole(
                            params= self.bl_q1_params, 
                            q_label= "q1", q_unit= "rad",
                            c_label= "c1", c_unit= "rad")

        self.BL_q2 = BacklashSinkhole(
                            params= self.bl_q2_params, 
                            q_label= "q2", q_unit= "rad",
                            c_label= "c2", c_unit= "rad")
        
        self.R = np.eye(3)
    
    @property
    def R(self)-> np.ndarray:
        return(self._R)
    
    @R.setter
    def R(self, R:np.ndarray) -> None:
        self._R = deepcopy(R)

    @property
    def ang_jac_q1(self) -> np.ndarray:
        return(self._ang_jac_q1)

    @ang_jac_q1.setter
    def ang_jac_q1(self, ang_jac : np.ndarray) -> None:
        self._ang_jac_q1 = deepcopy(ang_jac)


    @property
    def ang_jac_q2(self) -> np.ndarray:
        return(self._ang_jac_q2)

    @ang_jac_q2.setter
    def ang_jac_q2(self, ang_jac : np.ndarray) -> None :
        self._ang_jac_q2 = deepcopy(ang_jac)

    def func_jacobian_angular_vel (self, 
                                   q1_k : np.ndarray, q1_k_1 : np.ndarray,
                                   q2_k : np.ndarray, q2_k_1 : np.ndarray) -> np.ndarray:
        f1 = self.BL_q1.evaluate(q1_k, q1_k_1)
        f2 = self.BL_q2.evaluate(q2_k, q2_k_1)

        return(lambda q1_k, q1_k_1, q2_k, q2_k_1 : f1(q1_k, q1_k_1) * self.ang_jac_q1 + 
                                                    f2(q2_k, q2_k_1) * self.ang_jac_q2 )
    
    def R_dt (self, q1_k : np.ndarray, q1_k_1 : np.ndarray,
                    q2_k : np.ndarray, q2_k_1 : np.ndarray) -> np.ndarray :
        '''
        Compute the relative Rotation between two timesteps resulting from actuator motion
        
        Uses the Exponential map of the SO3 manifold to project the angular velocity onto the manifold of rotations in 3D
        
        Returns
        -------
        R_dt : ndarray (3x3)
            the relative rotation between timestep k and k-1
        '''
        f =  self.func_jacobian_angular_vel(q1_k, q1_k_1, q2_k, q2_k_1)
        return (SO3.exp (f(q1_k, q1_k_1, q2_k, q2_k_1)))
    
    def f (self, q1_k : np.ndarray, q1_k_1 : np.ndarray,
                 q2_k : np.ndarray, q2_k_1 : np.ndarray) -> None :
        """
        State transition function
        R(k) = R(k-1) @ R_dt

        Parameters
        ----------
        q1_k :    actuator 1 position at time k
        q1_k_1 :  actuator 1 position at time k-1
        q2_k :    actuator 2 position at time k
        q2_k_1 :  actuator 2 position at time k-1
        """
        self.R = self.R @ self.R_dt(q1_k, q1_k_1, q2_k, q2_k_1)

from particles import state_space_models as ssm
from scipy.spatial.transform import Rotation as R

from utils.uniform_axis_von_mises_spin_distribution import UARS_von_Mises

class EndoscopeMotionModel_SSM(ssm.StateSpaceModel):
    '''
    Probabilistic Motion model of a two bending plane endoscope to be used with
    the Particles library in order to 
        - simulate a particle filter based state estimation
        - learn the motion model parameters using SMC2
    
    

    References
    ----------
    Particles : https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/index.html

    SMC2 : 
        REF : SMC^2: an efficient algorithm for sequential analysis of state-space models

        Authors : Chopin et al.

        DOI : 10.48550/arXiv.1101.1528
    '''
    
    default_params = {'mq1' : 1, 'wq1' : 0.04, 'mq2': 1, 'wq2' : 0.3,           # Backlash model params for act 1 and 2
                        'Jwx_q1' : -0.24 , 'Jwy_q1' : 0.95, 'Jwz_q1': 0.11,     # Angular Jacobian for act 1
                        'Jwx_q2' : -0.78, 'Jwy_q2' : -0.01, 'Jwz_q2': 0.1,      # Angular Jacobian for act 2
                        'kappa_x0' : 100, 'kappa_x' : 1000, 'kappa_y' : 100}    # concentration for the UARS noise distribution in SO3
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        robot_params = np.array([self.mq1, self.wq1,                            # Backlash model params for act 1
                                self.mq2, self.wq2,                             # Backlash model params for act 2
                                self.Jwx_q1, self.Jwy_q1, self.Jwz_q1,          # Angular Jacobian for act 1
                                self.Jwx_q2, self.Jwy_q2, self.Jwz_q2])         # Angular Jacobian for act 2
        self.endoscope_motion_model = EndoscopeMotionModel(robot_params)
    
    @classmethod
    def Set_Actuation(cls, q1, q2):
        '''
        Parameters
        ----------
        q1 : ndarray
            the displacement trajectory of actuator 1
        
        q2 : ndarray
            the displacement trajectory of actuator 2
        '''
        cls._q1 = q1
        cls._q2 = q2
    
    @property
    def q1(self):
        return type(self)._q1

    @property
    def q2(self):
        return type(self)._q2

    def PX0(self, starting_point = np.eye(3)): # Distribution of X_0
        return UARS_von_Mises(mode = starting_point, kappa= self.kappa_x0)
    
    def PX(self, t, xp): # Distribution of X_t given X_{t-1}=xp (p=past)
        try:
            q1_k = self.q1[t]
            q1_k_1 = self.q1[t-1]
            q2_k = self.q2[t]
            q2_k_1 = self.q2[t-1]
        except:
            raise Exception("could not load the trajectory of actuators, please set them up with Set_Actuation()")
        R_dt = self.endoscope_motion_model.R_dt(q1_k= q1_k, q1_k_1= q1_k_1, q2_k= q2_k, q2_k_1= q2_k_1)
        return UARS_von_Mises(mode= R.from_rotvec(xp).as_matrix() @ R_dt, kappa= self.kappa_x)
    
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return UARS_von_Mises(mode= x, kappa= self.kappa_y)

    def proposal0(self, data, t = None):
        if t is None : 
            return(self.PX0())
        else :
            # allows us to start the pf at a specific timestep
            return(self.PX0(starting_point=data[t]))
            

    def proposal(self, t, xp, data):
        try:
            q1_k = self.q1[t]
            q1_k_1 = self.q1[t-1]
            q2_k = self.q2[t]
            q2_k_1 = self.q2[t-1]
        except:
            raise Exception("could not load the trajectory of actuators, please set them up with Set_Actuation()")
        R_dt = self.endoscope_motion_model.R_dt(q1_k= q1_k, q1_k_1= q1_k_1, q2_k= q2_k, q2_k_1= q2_k_1)

        # to guide the particles, we reduce the error between each particle and the previous data point y[t-1]
        # and then move them forward using the endoscope motion model
        dR_y_x = np.transpose(R.from_rotvec(xp).as_matrix(), (0,2,1)) @ data[t-1]
        dR_y_x_halfway = R.from_matrix(dR_y_x).as_rotvec()/2
        
        return UARS_von_Mises(mode= R.from_rotvec(xp).as_matrix() @ R.from_rotvec(dR_y_x_halfway).as_matrix() @ R_dt, kappa= self.kappa_x)