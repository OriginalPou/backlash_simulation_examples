# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 28/05/2024

Identify the parameters of the two bending plane endoscope motion model
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from copy import deepcopy
from ukfm import SO3

import pyswarms as ps

from bayesian_backlash_model.models.endoscope_motion_model import EndoscopeMotionModel

from bayesian_backlash_model.utils.geodesic_distance import geodesic_distance

def check_model_behavior(model_params : np.ndarray, 
                         q1 : np.ndarray,
                         q2 : np.ndarray,
                         tip_rot : np.ndarray):
    
    endoscope = EndoscopeMotionModel(params= model_params)
    endoscope.R = tip_rot[:,:,0]

    robot_R_traj = np.zeros((3,3, len(q1)))
    robot_R_traj[:,:,0] = deepcopy(endoscope.R)
    for i in range(1, len(q1)):
        endoscope.f(q1_k = q1[i], q1_k_1 = q1[i-1], q2_k = q2[i], q2_k_1 = q2[i-1])
        robot_R_traj[:,:,i] = deepcopy(endoscope.R)

    rpy_sim = np.zeros((len(q1),3))
    rpy_model = np.zeros((len(q1),3))
    for i in range(len(q1)):
        rpy_sim[i,:]   = SO3.to_rpy(tip_rot[:,:,i])
        rpy_model[i,:] = SO3.to_rpy(robot_R_traj[:,:,i])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(rpy_sim[:,0], 'r', label = r'$roll_{sim}$')
    ax1.plot(rpy_sim[:,1], 'g', label = r'$pitch_{sim}$')
    ax1.plot(rpy_sim[:,2], 'b', label = r'$yaw_{sim}$')
    ax1.plot(rpy_model[:,0], 'r--', label = r'$roll_{model}$')
    ax1.plot(rpy_model[:,1], 'g--', label = r'$pitch_{model}$')
    ax1.plot(rpy_model[:,2], 'b--', label = r'$yaw_{model}$')
    ax1.grid()
    ax1.set_ylabel('robot tip rot.\n(rad)')
    ax1.set_xlabel('time(steps)')
    ax1.legend(frameon = True, loc = 'center left', bbox_to_anchor= (1, 0.5))

    '''
    check angular velocity
    ''' 
    # reinitialize endoscope attitude
    endoscope.R = tip_rot[:,:,0]
    ang_vel_sim = np.zeros((len(q1),3))
    ang_vel_mod = np.zeros((len(q1),3))
    for i in range(1, len(q1)):
        J = endoscope.func_jacobian_angular_vel(q1_k = q1[i], q1_k_1 = q1[i-1], q2_k = q2[i], q2_k_1 = q2[i-1])
        ang_vel_mod[i,:] = J(q1_k = q1[i], q1_k_1 = q1[i-1], q2_k = q2[i], q2_k_1 = q2[i-1])
        ang_vel_sim[i,:] = SO3.log(np.transpose(tip_rot[:,:,i-1])@tip_rot[:,:,i])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(ang_vel_sim[:,0], 'r', label = r'$\Omega x_{sim}$')
    ax1.plot(ang_vel_sim[:,1], 'g', label = r'$\Omega y_{sim}$')
    ax1.plot(ang_vel_sim[:,2], 'b', label = r'$\Omega z_{sim}$')
    ax1.plot(ang_vel_mod[:,0], 'r--', label = r'$\Omega x_{model}$')
    ax1.plot(ang_vel_mod[:,1], 'g--', label = r'$\Omega y_{model}$')
    ax1.plot(ang_vel_mod[:,2], 'b--', label = r'$\Omega z_{model}$')
    ax1.grid()
    ax1.set_ylabel('robot tip ang vel.\n(rad/s)')
    ax1.set_xlabel('time(steps)')
    ax1.legend(frameon = True, loc = 'center left', bbox_to_anchor= (1, 0.5))

def model_error (model_params : np.ndarray, 
                    q1 : np.ndarray,
                    q2 : np.ndarray,
                    tip_rot : np.ndarray) -> np.ndarray :
    '''
    Computes the sum of geodesic distances between 
        - the rotation between time step k and k-1 in the simulated data
        - the rotation between time step k and k-1 computed by the model

    Parameters
    ----------
    model_params : ndarray
        model parameters of the endoscope motion model
    q1 : ndarray
        actuator 1 displacement used to obtain the simulated trajectory of the robot
    q2 : ndarray
        actuator 2 displacement used to obtain the simulated trajectory of the robot
    tip_rot : ndarray
        the rotation trajectory of the endoscope tip obtained by the sim
    
    Returns
    ---------
    geodesic_distance_sum : ndarray
        the sum of the geodesic distance
    '''
    endoscope = EndoscopeMotionModel(params= model_params)
    # geodesic_distance_sum = 0
    angular_vel_error = 0
    for i in range(1, len(q1)):
        J = endoscope.func_jacobian_angular_vel(q1_k = q1[i], q1_k_1 = q1[i-1], q2_k = q2[i], q2_k_1 = q2[i-1])
        angular_vel_model = J(q1_k = q1[i], q1_k_1 = q1[i-1], q2_k = q2[i], q2_k_1 = q2[i-1])
        angular_vel_sim = SO3.log(np.transpose(tip_rot[:,:,i-1])@tip_rot[:,:,i])
        # R_dt_model = endoscope.R_dt(q1_k = q1[i], q1_k_1 = q1[i-1], q2_k = q2[i], q2_k_1 = q2[i-1])
        # R_dt_sim   = np.transpose(tip_rot[:,:,i-1])@tip_rot[:,:,i]
        # geodesic_distance_sum += geodesic_distance(R1= R_dt_model, R2= R_dt_sim)
        angular_vel_error += np.linalg.norm(angular_vel_model-angular_vel_sim)**2
    return(angular_vel_error)

def model_error_pso (model_params : np.ndarray, 
                    q1 : np.ndarray,
                    q2 : np.ndarray,
                    tip_rot : np.ndarray) -> np.ndarray :
    """
    Compute the endoscope motion model error for the particle swarm optimization algorithm
    
    Parameters
    ----------
    model_params : ndarray
        model parameters of the endoscope motion model associated with the swarm
    q1 : ndarray
        actuator 1 displacement used to obtain the simulated trajectory of the robot
    q2 : ndarray
        actuator 2 displacement used to obtain the simulated trajectory of the robot
    tip_rot : ndarray
        the rotation trajectory of the endoscope tip obtained by the sim
    
    Returns
    ---------
    geodesic_distance_sum : ndarray
        the sum of the geodesic distance for each of the particles
    """
    geodesic_distances = np.zeros((model_params.shape[0]))
    for i in range(model_params.shape[0]):
        geodesic_distances[i] = model_error(model_params= model_params[i,:], 
                                            q1= q1, q2=q2, tip_rot= tip_rot)
    return(geodesic_distances)

#%%
if __name__ == "__main__" :
    # load the evolution of the rotation of the tip when the bending part is considered (with Backlash)
    tip_rot_hyst = scipy.io.loadmat('data/tip_rot_hyst.mat')['tip_rot_hyst']
    # load the displacement of the actuators (in rad)
    q1 = scipy.io.loadmat('data/q1_Jac.mat')['q1']
    q2 = scipy.io.loadmat('data/q2_Jac.mat')['q2']

    robot_params = np.array([1, 0.01,                 # Backlash model params for act 1
                             1, 0.01,                 # Backlash model params for act 2
                             -0.05, 0.9, 0.,         # Angular Jacobian for act 1
                             -0.89, -0.03, 0.03])    # Angular Jacobian for act 2
    print(model_error(model_params= robot_params, q1= q1, q2= q2, tip_rot= tip_rot_hyst))
    # %%
    # Create bounds
    # max_bound = np.array([3, 0.5,                   # Backlash model params for act 1
    #                       3, 0.5,                   # Backlash model params for act 2
    #                       0.5, 3, 3,              # Angular Jacobian for act 1
    #                       3, 0.5, 3])             # Angular Jacobian for act 2
    
    # min_bound = np.array([0, 1e-5,                  # Backlash model params for act 1
    #                       0, 1e-5,                  # Backlash model params for act 2
    #                       -0.5, -3, -3,           # Angular Jacobian for act 1
    #                       -3, -0.5, -3])          # Angular Jacobian for act 2
    # bounds = (min_bound, max_bound)
    # # Set-up hyperparameters for the PSO algorithm
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

    # # Call instance of PSO
    # optimizer = ps.single.LocalBestPSO(n_particles=20, dimensions=10, options=options, bounds= bounds)
    # # run optimization
    # cost, pos = optimizer.optimize(model_error_pso, iters = 500, q1=q1, q2=q2, tip_rot=tip_rot_hyst)
    # %%
    pos = np.array([0.99587858,  0.04411124,  1.62349233,  0.30253271, -0.23885224,
        0.95432228,  0.11448884, -0.78147181, -0.0090405 , -0.10072446])
    check_model_behavior(model_params= pos, q1= q1, q2=q2, tip_rot= tip_rot_hyst)
    plt.show()
    # %%
    # model_error(model_params= pos, q1= q1, q2= q2, tip_rot= tip_rot_hyst)

# %%
