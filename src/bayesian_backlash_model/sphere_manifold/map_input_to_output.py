from ukfm import SO3
from backlash_simulation_examples.models.model_endoscope_two_bending_3D import EndoscopeTwoBendingModel

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__" :
    model = EndoscopeTwoBendingModel()
    actuator_limit = np.pi/2 * model.D_

    radius = 0.0754152
    sphere_center = np.array([0, 0, 0.02497])

    Rot = np.eye(3)
    t = np.array([0,0,1])

    R_cc = model.R_base_to_tip(np.array([0, 0]))
    t_cc = model.t_base_to_tip(np.array([0, 0]))
    # move one of the cables and compute state after using the SO(3) exp map
    q_var_q1 = np.linspace(0, actuator_limit/2, 100)
    
    from fit_sphere_to_workspace import geodesic_distance

    for i in range (q_var_q1.shape[0]):
        new_Rcc = model.R_base_to_tip(np.array([q_var_q1[i], 0]))
        angle_Z = - model.camera_model_params_["rotation_camera_wrt_bearing"]
        mat_rotation_camera_wrt_bearing = np.array([
            [np.cos(angle_Z), -np.sin(angle_Z), 0.0],
            [np.sin(angle_Z),  np.cos(angle_Z), 0.0],
            [0.0,              0.0,             1.0]
        ])
        
        
        
        tangent_vector = SO3.log(new_Rcc @ np.transpose(R_cc))
        #print(tangent_vector)
        
        angle = geodesic_distance(new_Rcc, R_cc)
        #print(angle)
        R_cc = new_Rcc
        t_cc = model.t_base_to_tip(np.array([q_var_q1[i], 0]))
        t_cc = (t_cc).reshape((-1,))

        Rot = Rot.dot(SO3.exp(np.array([tangent_vector[0]/2,tangent_vector[1]/2, 0])))
        new_t = (Rot @ t)*radius + sphere_center
        print(np.linalg.norm(new_t-t_cc))

    from plot_ref_frames import plot_frame
    from fit_sphere_to_workspace import draw_sphere

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.scatter(0,0,0, label="base")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    draw_sphere(ax, sphere_center, radius)

    TR_44_Sphere = np.eye(4)
    TR_44_Sphere[:-1, -1] = new_t.reshape((-1,))
    TR_44_Sphere[:-1, :-1] = Rot
    plot_frame(ax, TR_44 = TR_44_Sphere, frame_name = '', arrows_length = 0.05)
    
    TR_44_Sphere = np.eye(4)
    TR_44_Sphere[:-1, -1] = t_cc
    TR_44_Sphere[:-1, :-1] = R_cc
    plot_frame(ax, TR_44 = TR_44_Sphere, frame_name = '', arrows_length = 0.05)
    ax.plot([0, t_cc[0]], [0, t_cc[1]], [0, t_cc[2]], '--')

    ax.legend()
    plt.show()

    # p = np.zeros((q_var_q1.shape[0]*q_var_q2.shape[0],3))
    # for i in range(q_var_q1.shape[0]):
    #     for j in range(q_var_q2.shape[0]):
    #         q_values = np.zeros((1,2))
    #         q_values[:,0] = q_var_q1[i]
    #         q_values[:,1] = q_var_q2[j]
    #         p[i*q_var_q1.shape[0] + j,:] = model.t_base_to_tip(q_values.reshape((-1,1))).reshape((-1,))
    

