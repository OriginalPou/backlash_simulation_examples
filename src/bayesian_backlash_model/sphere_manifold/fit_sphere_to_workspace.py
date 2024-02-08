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
from scipy.spatial.transform import Rotation as R

def draw_sphere(ax, sphere_center, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + sphere_center[0]
    y = radius * np.sin(u) * np.sin(v) + sphere_center[1]
    z = radius * np.cos(v) + sphere_center[2]
    return ax.plot_wireframe(x, y, z, color="y")

def geodesic_distance(R_cc : np.ndarray, R_sph_n : np.ndarray):
    return(np.abs(np.arccos((np.trace(np.transpose(R_cc) @ R_sph_n)-1)/2)))
    

if __name__ == "__main__" :
    model = EndoscopeTwoBendingModel()
   
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.scatter(0,0,0, label="base")

    actuator_limit = np.pi/2 * model.D_

    q_var_q1 = np.linspace(-actuator_limit/2, actuator_limit/2, 100)
    q_var_q2 = np.linspace(-actuator_limit/2, actuator_limit/2, 100)

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
    print(radius)
    print(sphere_center)
    
    def translation_mean_error_sphere(points : np.ndarray, sphere_center : np.ndarray, radius):
        error = 0
        for i in range(points.shape[0]):
            error += np.abs(np.linalg.norm(points[i,:]-sphere_center) - radius)
        return(error/points.shape[0])

    mean_error = translation_mean_error_sphere(p, sphere_center, radius)
    print(mean_error)
    # Draw sphere
    
    
    draw_sphere(ax, sphere_center, radius)
    ax.set_title("Fitting a sphere to workspace with mean error of %.4f meters" % mean_error)

    ax.legend()
    # plt.show()

    """
    Now that we checked the translation error between the two models
    we need to check the rotation error
    On the sphere, the rotation of the camera equals the rotation around
    the normal of the sphere with model.camera_model_params_["rotation_camera_wrt_bearing"] degrees
    """
    
    q1 = actuator_limit/4
    q2 = actuator_limit/4
    # get the position of the tip of the endoscope
    p = model.t_base_to_tip(np.array([q1, q2]).reshape((-1,1))).reshape((-1,))
    # get its projection on the sphere
    def project_point_on_sphere(p, sphere_center, radius):
        # REF: https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
        vector_norm = np.linalg.norm(p - sphere_center)
        # scale the vector so that it has length equal to the radius of the sphere
        q = radius / vector_norm *  (p - sphere_center)
        # return the projection
        return(q + sphere_center)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(p[0],p[1],p[2], label="tip position")

    proj_p = project_point_on_sphere(p, sphere_center, radius)
    ax.scatter(proj_p[0],proj_p[1],proj_p[2], label="projection")
    draw_sphere(ax, sphere_center, radius)

    R_p = model.R_base_to_tip(np.array([q1, q2]))
    print(R_p)
    # get the normal to the sphere
    normal_proj_p = (proj_p - sphere_center)/radius 
    # R_proj_p = R.from_rotvec(normal_proj_p)
    # print(R_proj_p.as_matrix())
    
    # REF: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    R_proj_p = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_proj_p)
    # Add rotation of camera w.r.t to bearing
    angle_Z = model.camera_model_params_["rotation_camera_wrt_bearing"]
    mat_rotation_camera_wrt_bearing = np.array([
            [np.cos(angle_Z), -np.sin(angle_Z), 0.0],
            [np.sin(angle_Z),  np.cos(angle_Z), 0.0],
            [0.0,              0.0,             1.0]
        ])
    R_proj_p = R_proj_p @ mat_rotation_camera_wrt_bearing

    from plot_ref_frames import plot_frame
    
    def plot_robot(ax, c) :
        #pts = model.get_wireframe(c)
        #ax.plot(pts[:,0], pts[:,1], pts[:,2], c=color, label=label)
        label="_dummy"
        wireframe_per_body = model.get_wireframe_per_body(c)
        ax.plot(wireframe_per_body['body'][:,0],
                wireframe_per_body['body'][:,1],
                wireframe_per_body['body'][:,2],
                '--',
                c='k', label=label)
        ax.plot(wireframe_per_body['flexible'][:,0],
                wireframe_per_body['flexible'][:,1],
                wireframe_per_body['flexible'][:,2],
                '--',
                c='g', label="_"+label)
        ax.plot(wireframe_per_body['tip'][:,0],
                wireframe_per_body['tip'][:,1],
                wireframe_per_body['tip'][:,2],
                '--',
                c='k', label="_"+label)
        ax.scatter(wireframe_per_body['tip'][-1,0],
                    wireframe_per_body['tip'][-1,1],
                    wireframe_per_body['tip'][-1,2],
                    c='r', label="_"+label)

    # plot the pose from the constant curvature model
    TR_44_Fc = np.eye(4)
    TR_44_Fc[:-1, -1] = p.reshape((-1,))
    TR_44_Fc[:-1, :-1] = R_p
    plot_robot(ax, np.array([q1, q2]))
    plot_frame(ax, TR_44 = TR_44_Fc, frame_name = '', arrows_length = 0.05)

    # plot the pose normal to the sphere manifold 
    TR_44_Sphere = np.eye(4)
    TR_44_Sphere[:-1, -1] = proj_p.reshape((-1,))
    TR_44_Sphere[:-1, :-1] = R_proj_p
    plot_frame(ax, TR_44 = TR_44_Sphere, frame_name = '', arrows_length = 0.05)

    ax.scatter(sphere_center[0],sphere_center[1],sphere_center[2], label="sphere center")
    ax.plot([sphere_center[0], proj_p[0]], [sphere_center[1], proj_p[1]], [sphere_center[2], proj_p[2]], '--')

    ax.legend()
    plt.show()

    # compute the rotation error using the Angular (geodesic) distance in SO(3)
    
    # r = R.from_matrix(np.transpose(R_p) @ R_proj_p)
    # angle = np.linalg.norm(r.as_rotvec())
    # axis = r.as_rotvec()/angle
    # print(angle)
    # print(axis)

    print(geodesic_distance(R_p, R_proj_p))
