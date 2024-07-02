# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 27/05/2024

Analyze data generated from the Tendon Actuated Continuum Robot simulator
"""

#%%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from ukfm import SO3

def draw_sphere(ax, sphere_center, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + sphere_center[0]
    y = radius * np.sin(u) * np.sin(v) + sphere_center[1]
    z = radius * np.cos(v) + sphere_center[2]
    return ax.plot_wireframe(x, y, z, color="grey")

def angular_jacobian(tip_rot : np.ndarray, q : np.ndarray) -> np.ndarray:
    '''
    Compute the angular Jacobian J_{\Omega}(q1)
    i.e the rate of change of the attitude of the tip wrt the rate of change of the actuator displacement
    \Omega = J_q * \dot{q}
    J_q = dR/dq = (dR/dt)/(dq/dt)

    Parameters
    ----------
        tip_rot : ndarray
            the attitude of the tip
        q : ndarray
            the displacement of the actuator

    Returns
    --------
        ang_jac_q : ndarray
            the angular Jacobian w.r.t q
    '''
    # Comptute the angular velocity
    ang_vel_q = np.zeros((3, len(q)))
    for t in range(1, len(q)):
        '''
        rotation of the tip between timestep t and t-1
        R^{t-1}_{t} = (R^{w}_{t-1})^T @ R^{w}_{t}
        The attitude of the tip at timestep t w.r.t the attitude of the tip at t-1
        '''
        ang_vel_q[:,t] = SO3.log(np.transpose(tip_rot[:,:,t-1])@tip_rot[:,:,t])
    
    # compute the change in motor displacement
    dq1 = np.zeros((len(q1),1))
    dq1[1:] = q1[1:] - q1[:-1]
    dq1 = dq1 + 10e-15 # avoid dividing by zero

    # compute the angular Jacobian wrt q1
    ang_jac_q = np.zeros((3, len(q)))
    ang_jac_q[0,:] = np.divide(ang_vel_q[0,:],dq1.T)
    ang_jac_q[1,:] = np.divide(ang_vel_q[1,:],dq1.T)
    ang_jac_q[2,:] = np.divide(ang_vel_q[2,:],dq1.T)

    return(ang_jac_q)
#%%
if __name__ == "__main__" :

    ## Compute the Angular Jacobian w.r.t the actuator 1
    # load the tip position and rotation when only the tendons 1-3 are actuated
    tip_pos_q1 = scipy.io.loadmat('data/tip_pos_q1.mat')['tip_pos']
    tip_rot_q1 = scipy.io.loadmat('data/tip_rot_q1.mat')['tip_rot']
    # load the trajectory of actuator 1 (in rad)
    q1 = scipy.io.loadmat('data/q1_Jac.mat')['q1']
    
    # Compute the angular Jacobian w.r.t q1 when the bending part is not considered (no hysteresis)
    ang_jac_q1 = angular_jacobian(tip_rot= tip_rot_q1, q= q1)

    # compute the mean and std of the angular Jacobian w.r.t q1
    print("std = " + str(np.round(np.std(ang_jac_q1,axis = 1),2)))
    print("mean = " + str(np.round(np.mean(ang_jac_q1,axis = 1),2)))


    #%% Compute the Angular Jacobian w.r.t the actuator 2
    # load the tip position and rotation when only the tendons 2-4 are actuated
    tip_pos_q2 = scipy.io.loadmat('data/tip_pos_q2.mat')['tip_pos']
    tip_rot_q2 = scipy.io.loadmat('data/tip_rot_q2.mat')['tip_rot']
    # load the trajectory of actuator 2 (in rad)
    q2 = scipy.io.loadmat('data/q2_Jac.mat')['q2']
    
    # Compute the angular Jacobian w.r.t q2 when the bending part is not considered (no hysteresis)
    ang_jac_q2 = angular_jacobian(tip_rot= tip_rot_q2, q= q2)

    # compute the mean and std of the angular Jacobian w.r.t q2
    print("std = " + str(np.round(np.std(ang_jac_q2,axis = 1),2)))
    print("mean = " + str(np.round(np.mean(ang_jac_q2,axis = 1),2)))
    
    #%%
    plt.figure()
    plt.plot(ang_jac_q1[1,:])
    #plt.plot(dq1)
    #plt.plot(ang_jac_q1[0,:])
    # %%
    # tip_pos_hyst = scipy.io.loadmat('data/tip_pos_hyst.mat')['tip_pos_hyst']
    # tip_rot_hyst = scipy.io.loadmat('data/tip_rot_hyst.mat')['tip_rot_hyst']

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_box_aspect([1,1,1])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # draw_sphere(ax, [0,0,0], 1)

    # plt.show()
