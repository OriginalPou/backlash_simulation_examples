# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on Friday Feb 02 2024

@author: Mahdi Chaari
"""

from backlash_simulation_examples.models.model_endoscope_two_bending_3D import EndoscopeTwoBendingModel

from utils.fitting import sphere_fit

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__" :
    model = EndoscopeTwoBendingModel()
   
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.scatter(0,0,0, label="base")

    actuator_limit = np.pi/2 * model.D_

    q_var_q1 = np.linspace(-actuator_limit, actuator_limit, 100)
    q_var_q2 = np.linspace(-actuator_limit, actuator_limit, 100)

    p = np.zeros((q_var_q1.shape[0]*q_var_q2.shape[0],3))
    for i in range(q_var_q1.shape[0]):
        for j in range(q_var_q2.shape[0]):
            q_values = np.zeros((1,2))
            q_values[:,0] = q_var_q1[i]
            q_values[:,1] = q_var_q2[j]
            p[i*q_var_q1.shape[0] + j,:] = model.t_base_to_tip(q_values.reshape((-1,1))).reshape((-1,))
    
    ax.scatter(p[:,0], p[:,1], p[:,2],  label="tip positions")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ## Fitting a sphere to the workspace
    radius, sphere_center = sphere_fit(p)
    
    def mean_error_sphere(points : np.ndarray, sphere_center, radius):
        error = 0
        for i in range(points.shape[0]):
            error += np.abs(np.linalg.norm(points[i,:]-sphere_center) - radius)
        return(error/points.shape[0])

    mean_error = mean_error_sphere(p, sphere_center, radius)
    print(mean_error)
    ## Draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + sphere_center[0]
    y = radius * np.sin(u) * np.sin(v) + sphere_center[1]
    z = radius * np.cos(v) + sphere_center[2]
    ax.plot_wireframe(x, y, z, color="r")

    ax.legend()
    plt.show()
