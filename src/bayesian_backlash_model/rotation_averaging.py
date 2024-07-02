#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import gtsam
from gtsam.utils import plot
from gtsam import Rot3, Point3, Pose3

from ukfm import SO3

from utils.uniform_axis_von_mises_spin_distribution import UARS_von_Mises
from utils.geodesic_distance import geodesic_distance




if __name__ == "__main__" :
    # load the evolution of the rotation of the tip when the bending part is considered (with Backlash)
    tip_rot_hyst = scipy.io.loadmat('data/tip_rot_hyst_speed_ps.mat')['tip_rot_hyst']    
    
    # initialize graph and initial values
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    initial.insert(0,  gtsam.SO3(tip_rot_hyst[:,:,0]))
    # initialize noise model
    NOISE_MODEL = gtsam.noiseModel.Isotropic.Sigma(dim=3, sigma=np.deg2rad(5))
    # set prior
    prior = gtsam.PriorFactorSO3(0, gtsam.SO3(tip_rot_hyst[:,:,0]), gtsam.noiseModel.Isotropic.Sigma(dim=3, sigma=np.deg2rad(1)))
    graph.push_back(prior)
    # add odometry factors
    # each node is linked to the next three
    for i in range (tip_rot_hyst.shape[2]):
        for j in range(1,7):
            if i + j < tip_rot_hyst.shape[2]:
                dR_i_j = np.transpose(tip_rot_hyst[:,:,i])@tip_rot_hyst[:,:,i+j]
                R_noise = UARS_von_Mises(kappa=200, mode= dR_i_j).rvs()
                graph.add(gtsam.FrobeniusBetweenFactorSO3(i, i+j, gtsam.SO3(R_noise), NOISE_MODEL))
                if j == 1 :
                    initial.insert(i+1, gtsam.SO3(initial.atSO3(i).matrix()@ R_noise))
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    #%%
    # check for loop closures
    loop = 0
    for i in range(tip_rot_hyst.shape[2]-1):
        for j in range(i+5, tip_rot_hyst.shape[2]-4):
            dR_i_j = np.transpose(tip_rot_hyst[:,:,i])@tip_rot_hyst[:,:,j]
            R_noise = UARS_von_Mises(kappa=200, mode= dR_i_j).rvs()
            if geodesic_distance(R_noise, np.eye(3))<1e-2:
                graph.add(gtsam.FrobeniusBetweenFactorSO3(i, j, gtsam.SO3(R_noise), NOISE_MODEL))
                loop += 1
    print(loop)
    #%%
    #graph.saveGraph("EndoscopeRotationGraphLoops.dot", initial)
    result_lc = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    #%%
    rpy_nois_ = np.zeros((tip_rot_hyst.shape[2],3))
    rpy_opti_lc_ = np.zeros((tip_rot_hyst.shape[2],3))
    for i in range(tip_rot_hyst.shape[2]):
        rpy_nois_[i,:] = SO3.to_rpy(initial.atSO3(i).matrix())
        rpy_opti_lc_[i,:] = SO3.to_rpy(result_lc.atSO3(i).matrix())
    
    # %%
    rpy_opti = np.zeros((tip_rot_hyst.shape[2],3))
    rpy_nois = np.zeros((tip_rot_hyst.shape[2],3))
    rpy_sim = np.zeros((tip_rot_hyst.shape[2],3))
    rpy_opti_lc = np.zeros((tip_rot_hyst.shape[2],3))
    for i in range(tip_rot_hyst.shape[2]):
        rpy_sim[i,:]   = SO3.to_rpy(tip_rot_hyst[:,:,i])
        rpy_nois[i,:] = SO3.to_rpy(initial.atSO3(i).matrix())
        rpy_opti[i,:] = SO3.to_rpy(result.atSO3(i).matrix())
        rpy_opti_lc[i,:] = SO3.to_rpy(result_lc.atSO3(i).matrix())

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(rpy_sim[:,0], 'r', label = r'$roll_{sim data}$')
    ax1.plot(rpy_sim[:,1], 'g', label = r'$pitch_{sim data}$')
    ax1.plot(rpy_sim[:,2], 'b', label = r'$yaw_{sim data}$')
    #ax1.plot(rpy_opti_lc[:,0], 'r--', label = r'$roll_{rot avg}$')
    #ax1.plot(rpy_opti_lc[:,1], 'g--', label = r'$pitch_{rot avg}$')
    #ax1.plot(rpy_opti_lc[:,2], 'b--', label = r'$yaw_{rot avg}$')
    ax1.plot(rpy_opti[:,0], 'r--', label = r'$roll_{gtsam}$')
    ax1.plot(rpy_opti[:,1], 'g--', label = r'$pitch_{gtsam}$')
    ax1.plot(rpy_opti[:,2], 'b--', label = r'$yaw_{gtsam}$')
    ax1.grid()
    ax1.set_ylabel('robot tip rot.\n(rad)')
    ax1.set_xlabel('time(steps)')
    ax1.legend(frameon = True, loc = 'center left', bbox_to_anchor= (1, 0.5))
    plt.show()
    #%%
    geodesic_errors_opti_ = np.zeros(tip_rot_hyst.shape[2])
    geodesic_errors_opti_lp_ = np.zeros(tip_rot_hyst.shape[2])
    for i in range(tip_rot_hyst.shape[2]):
        geodesic_errors_opti_[i] = geodesic_distance(tip_rot_hyst[:,:,i], result.atSO3(i).matrix())
        geodesic_errors_opti_lp_[i] = geodesic_distance(tip_rot_hyst[:,:,i], result_lc.atSO3(i).matrix())
# %%
    geodesic_errors_opti = np.zeros(tip_rot_hyst.shape[2])
    geodesic_errors_noise = np.zeros(tip_rot_hyst.shape[2])
    geodesic_errors_opti_lp = np.zeros(tip_rot_hyst.shape[2])
    for i in range(tip_rot_hyst.shape[2]):
        geodesic_errors_noise[i] = geodesic_distance(tip_rot_hyst[:,:,i], initial.atSO3(i).matrix())
        geodesic_errors_opti[i] = geodesic_distance(tip_rot_hyst[:,:,i], result.atSO3(i).matrix())
        geodesic_errors_opti_lp[i] = geodesic_distance(tip_rot_hyst[:,:,i], result_lc.atSO3(i).matrix())
    #%%
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(geodesic_errors_noise, 'r', label = r'$error (1 neighbor)$')
    #ax1.plot(geodesic_errors_opti, 'b', label = r'$error_{rotavg} (10 neighbors)$')
    #ax1.plot(geodesic_errors_opti_lp, 'g', label = r'$error_{rotavg+lc}(10 neighbors)$')
    ax1.plot(geodesic_errors_opti_, 'b--', label = r'$error_{rotavg} (7 neighbors)$')
    ax1.plot(geodesic_errors_opti_lp_, 'g--', label = r'$error_{rotavg+lc}(7 neighbors)$')
    
    ax1.grid()
    ax1.set_ylabel('Chordal distance to ground truth.\n(rad)')
    ax1.set_xlabel('time(steps)')
    ax1.set_title("Endoscope state estimation error \n with rotation averaging and loop closures ")
    ax1.legend(loc = 'upper left')
    plt.show()
    fig.savefig('rot_avg_err.png', dpi = 300)
# %%
    marginals = gtsam.Marginals(graph, result)
    plot.plot_trajectory(1, result, marginals=marginals, scale=8)
# %%
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # pose = gtsam.Pose3(result.atRot3(1), np.array([0,0,0]))
    # plot_pose3_on_axes(ax, pose, P=marginals.marginalCovariance(0))
    # pose = gtsam.Pose3(result.atRot3(1), np.array([5,0,0]))
    # plot_pose3_on_axes(ax, pose, P=marginals.marginalCovariance(199))
    # plt.show()
# %%
