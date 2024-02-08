#%%
from ukfm import SO3
from backlash_simulation_examples.models.model_endoscope_two_bending_3D import EndoscopeTwoBendingModel

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm

#%%
if __name__ == "__main__" :
    model = EndoscopeTwoBendingModel()
    # print(model.func_jacobian_angular_vel_wrt_c_angle_(np.array([[0.3],[0.2]])))
    actuator_limit = np.pi/2 * model.D_

    radius = 0.0754152
    sphere_center = np.array([0, 0, 0.02497])

    Rot = np.eye(3)
    t = np.array([0,0,1])

    R_cc = model.R_base_to_tip(np.array([0, 0]))
    t_cc = model.t_base_to_tip(np.array([0, 0]))
    # move one of the cables and compute state after using the SO(3) exp map
    q_var_q1 = np.linspace(0, actuator_limit/2, 100)
    q_var_q2 = np.linspace(0, actuator_limit/2, 100)
    
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

        Rot = Rot.dot(SO3.exp(np.array([tangent_vector[0],tangent_vector[1], 0])))
        new_t = (Rot @ t)*radius + sphere_center
        #print(np.linalg.norm(new_t-t_cc))

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
    # plt.show()

    # p = np.zeros((q_var_q1.shape[0]*q_var_q2.shape[0],3))
    # for i in range(q_var_q1.shape[0]):
    #     for j in range(q_var_q2.shape[0]):
    #         q_values = np.zeros((1,2))
    #         q_values[:,0] = q_var_q1[i]
    #         q_values[:,1] = q_var_q2[j]
    #         p[i*q_var_q1.shape[0] + j,:] = model.t_base_to_tip(q_values.reshape((-1,1))).reshape((-1,))
    
    ## Experiment : From a random point on the workspace,
    ##              Compute the angle you move (or geodesic distance)
    ##              and the tangent vector to the SO(3) manifold

    q1 = -actuator_limit/10
    q2 = -actuator_limit/4

    q_var = np.linspace(-actuator_limit/4, actuator_limit/4, 25)

    
    q1_var = np.array([- actuator_limit/200, 0 , actuator_limit/200])
    q2_var = np.array([- actuator_limit/200, 0 , actuator_limit/200])

    for i in range(q_var.shape[0]-1):
        Rot_2 = model.R_base_to_tip(np.array([actuator_limit/10, q_var[i]]))
        Rot_dq2 = model.R_base_to_tip(np.array([actuator_limit/10, q_var[i+1]]))

        tangent_vector = SO3.log(np.transpose(Rot_2) @ Rot_dq2)
        #print("the tangent vector for dq2 = %.5f and q2 = %.5f" %(q_var[i+1] - q_var[i], q_var[i]))
        #print(tangent_vector)

        Rot_1 = model.R_base_to_tip(np.array([q_var[i], 0]))
        Rot_dq1 = model.R_base_to_tip(np.array([q_var[i], actuator_limit/20]))

        
        symmetry = geodesic_distance(Rot_dq2, Rot_2)-geodesic_distance(Rot_dq1, Rot_1)
        #print("%.5f" % symmetry)

    # q1 = -actuator_limit/4
    # q2 = 0
    # Rot = model.R_base_to_tip(np.array([q1, q2]))
    # q_var = np.linspace(0, actuator_limit/2, 100)
    # for i in range(q_var.shape[0]):
    #     Rot_dq = model.R_base_to_tip(np.array([q1 , q2 + q_var[1]]))
    #     print("the geodesic distance for dq2 = %.5f and q2 = %.5f" %(q_var[1], q2))
    #     print("%.5f" % geodesic_distance(Rot_dq, Rot))
    #     q2 = q2 + q_var[1]
    #     Rot = Rot_dq

    #%%    
    ## Experiment : Sample points on the workspace and compute the 
    ##              angular velocity jacobian w.r.t the cable actuation
    ##              Compute the mean and std of the jacobians

    # q_var_q1 = np.linspace(- actuator_limit/4, actuator_limit/4, 10000)
    # q_var_q2 = np.linspace(- actuator_limit/4, actuator_limit/4, 10000)

    # nb_samples = q_var_q1.shape[0]

    # ang_vel_jacobians_wrt_cables = np.zeros((3,2,nb_samples**2))

    # for i in tqdm(range (nb_samples)):
    #     for j in range(nb_samples):
    #         ang_vel_jacobians_wrt_cables[:,:,i*nb_samples + j] = \
    #             model.jacobian_angular(np.array([q_var_q1[j], q_var_q2[i]]))

    # print(np.mean(ang_vel_jacobians_wrt_cables, axis = 2))
    # print(np.std(ang_vel_jacobians_wrt_cables, axis = 2, ddof = 1)) 
    # print(np.max(ang_vel_jacobians_wrt_cables, axis = 2))  
    # print(np.min(ang_vel_jacobians_wrt_cables, axis = 2))

    """
    Mean = 
    [[-2.15095497e-18  1.61115794e+02]
    [ 5.55087276e+00  3.34694050e-18]
    [-2.76797437e-14  1.99775968e-15]]

    std =
    [[  5.50563895   4.92599069]
    [106.2251326  121.23648694]
    [ 36.02595182  36.02595182]]
    """    

    #%%
    ## Plot the angular velocity vector field
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])

    q_var_q1 = np.linspace(- actuator_limit/4, actuator_limit/4, 10)
    q_var_q2 = np.linspace(- actuator_limit/4, actuator_limit/4, 10)

    nb_samples = q_var_q1.shape[0]

    for i in range (nb_samples):
        for j in range(nb_samples):
            ang_vel_jacobians_wrt_cables = \
                model.jacobian_angular(np.array([q_var_q1[j], q_var_q2[i]]))
            lin_vel_jacobians_wrt_cables = \
                model.jacobian_position(np.array([q_var_q1[j], q_var_q2[i]]))
            
            pos = model.R_base_to_tip(np.array([q_var_q1[j], q_var_q2[i]])) @ t * radius + sphere_center
            
            ang_vel = ang_vel_jacobians_wrt_cables @ np.array([0, 0.001])
            lin_vel = lin_vel_jacobians_wrt_cables @ np.array([0, 0.001])
            # ax.quiver(pos[0], pos[1], pos[2], ang_vel[0], ang_vel[1], 0, \
            #           length = 0.08, color = 'black', arrow_length_ratio = 0.4)
            ax.quiver(pos[0], pos[1], pos[2], ang_vel[0], ang_vel[1], ang_vel[2], \
                      length = 0.08, color = 'black', arrow_length_ratio = 0.4)
            ax.quiver(pos[0], pos[1], pos[2], lin_vel[0], lin_vel[1], lin_vel[2], \
                      length = 0.5, color = 'red', arrow_length_ratio = 0.4)

    q_var_q1 = np.linspace(- actuator_limit/4, actuator_limit/4, 4)
    q_var_q2 = np.linspace(- actuator_limit/4, actuator_limit/4, 4)
    nb_samples = q_var_q1.shape[0]
    for i in range (nb_samples):
        for j in range(nb_samples):  
            TR_44 = np.eye(4)
            TR_44[:-1, -1] = model.t_base_to_tip(np.array([q_var_q1[i], q_var_q2[j]])).reshape((-1,))
            TR_44[:-1, :-1] = model.R_base_to_tip(np.array([q_var_q1[i], q_var_q2[j]]))
            plot_frame(ax, TR_44 = TR_44, frame_name = '', arrows_length = 0.01)

    draw_sphere(ax, sphere_center, radius)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
       

    #%%
    ## Plot the angular velocity vector field
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])

    q_var_q1 = np.linspace(- actuator_limit/4, actuator_limit/4, 10)
    q_var_q2 = np.linspace(- actuator_limit/4, actuator_limit/4, 10)

    nb_samples = q_var_q1.shape[0]

    for i in range (nb_samples):
        for j in range(nb_samples):
            ang_vel_jacobians_wrt_cables = \
                model.jacobian_angular(np.array([q_var_q1[j], q_var_q2[i]]))
            lin_vel_jacobians_wrt_cables = \
                model.jacobian_position(np.array([q_var_q1[j], q_var_q2[i]]))
            
            pos = model.R_base_to_tip(np.array([q_var_q1[j], q_var_q2[i]])) @ t * radius + sphere_center
            
            ang_vel = ang_vel_jacobians_wrt_cables @ np.array([0.001, 0])
            lin_vel = lin_vel_jacobians_wrt_cables @ np.array([0.001, 0])

            ax.quiver(pos[0], pos[1], pos[2], ang_vel[0], ang_vel[1], ang_vel[2], \
                      length = 0.08, color = 'black', arrow_length_ratio = 0.4)
            ax.quiver(pos[0], pos[1], pos[2], lin_vel[0], lin_vel[1], lin_vel[2], \
                      length = 0.5, color = 'red', arrow_length_ratio = 0.4)

    q_var_q1 = np.linspace(- actuator_limit/4, actuator_limit/4, 4)
    q_var_q2 = np.linspace(- actuator_limit/4, actuator_limit/4, 4)
    nb_samples = q_var_q1.shape[0]
    for i in range (nb_samples):
        for j in range(nb_samples):  
            TR_44 = np.eye(4)
            TR_44[:-1, -1] = model.t_base_to_tip(np.array([q_var_q1[i], q_var_q2[j]])).reshape((-1,))
            TR_44[:-1, :-1] = model.R_base_to_tip(np.array([q_var_q1[i], q_var_q2[j]]))
            plot_frame(ax, TR_44 = TR_44, frame_name = '', arrows_length = 0.01)

    draw_sphere(ax, sphere_center, radius)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
       
    plt.show()


# %%
    ## Experiment : compute the tangent vector of rotations
    ##              induced by small cable actuation

q_var_q1 = np.linspace(- actuator_limit/4, actuator_limit/4, 1000)
q_var_q2 = np.linspace(- actuator_limit/4, actuator_limit/4, 1000)

nb_samples = q_var_q1.shape[0]

ang_vel_jacobians_wrt_cables = np.zeros((3,1,nb_samples**2))

for i in tqdm(range (nb_samples)):
    for j in range(nb_samples):
        Rot = model.R_base_to_tip(np.array([q_var_q1[j], q_var_q2[i]]))
        Rot_dq = model.R_base_to_tip(np.array([q_var_q1[j] + 0.0001, q_var_q2[i]]))

        ang_vel_jacobians_wrt_cables[:,:,i*nb_samples + j] = SO3.log(Rot_dq @ np.transpose(Rot)).reshape((3,1))

    





# %%
print(np.mean(ang_vel_jacobians_wrt_cables, axis = 2))
print(np.std(ang_vel_jacobians_wrt_cables, axis = 2, ddof = 1))
# %%
ang_vel_jacobians_wrt_cables = np.zeros((3,1,nb_samples**2))

for i in tqdm(range (nb_samples)):
    for j in range(nb_samples):
        Rot = model.R_base_to_tip(np.array([q_var_q1[j], q_var_q2[i]]))
        Rot_dq = model.R_base_to_tip(np.array([q_var_q1[j] , q_var_q2[i]+ 0.001]))

        ang_vel_jacobians_wrt_cables[:,:,i*nb_samples + j] = SO3.log(Rot_dq @ np.transpose(Rot)).reshape((3,1))

    

# %%
print(np.mean(ang_vel_jacobians_wrt_cables, axis = 2))
print(np.std(ang_vel_jacobians_wrt_cables, axis = 2, ddof = 1))

# %%
