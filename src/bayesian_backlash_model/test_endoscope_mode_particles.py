# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 07/06/2024
"""
#%%
from models.endoscope_motion_model import EndoscopeMotionModel_SSM

from ukfm import SO3

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# load the trajectory of actuator 1 and 2 (in rad)
q1 = scipy.io.loadmat('data/q1_Jac.mat')['q1']
q2 = scipy.io.loadmat('data/q2_Jac.mat')['q2']

# load the evolution of the rotation of the tip when the bending part is considered (with Backlash)
tip_rot_hyst = scipy.io.loadmat('data/tip_rot_hyst.mat')['tip_rot_hyst']
tip_rot_hyst = np.transpose(tip_rot_hyst, (2, 0, 1))
    

EndoscopeMotionModel_SSM.Set_Actuation(q1=q1, q2=q2)


# %%
ssm = EndoscopeMotionModel_SSM(kappa_x = 1000, kappa_x0 = 100)
trajectory = ssm.simulate(T= len(q1))
rpy_traj = np.zeros((len(q1),3))
rpy_sim = np.zeros((len(q1),3))
for i in range (len(q1)):
    rpy_sim[i]   = SO3.to_rpy(tip_rot_hyst[i])
    rpy_traj[i]   = SO3.to_rpy(trajectory[0][i])
# %%
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(rpy_sim[:,0], 'r', label = r'$roll_{sim}$')
ax1.plot(rpy_sim[:,1], 'g', label = r'$pitch_{sim}$')
ax1.plot(rpy_sim[:,2], 'b', label = r'$yaw_{sim}$')
ax1.plot(rpy_traj[:,0], 'r--', label = r'$roll_{ssm}$')
ax1.plot(rpy_traj[:,1], 'g--', label = r'$pitch_{ssm}$')
ax1.plot(rpy_traj[:,2], 'b--', label = r'$yaw_{ssm}$')
ax1.grid()
ax1.set_ylabel('robot tip rot.\n(rad)')
ax1.set_xlabel('time(steps)')
ax1.legend(frameon = True, loc = 'center left', bbox_to_anchor= (1, 0.5))
plt.show()
# %%
from particles import mcmc
from particles import distributions as dists

prior_dict = {'wq1': dists.TruncNormal(a= 0, b = 1, mu = 0.5, sigma=1.),
              'wq2': dists.TruncNormal(a= 0, b = 1, mu = 0.5, sigma=1.),
              'Jwx_q1' : dists.Normal(loc = 0, scale= 0.5),
              'Jwy_q1' : dists.Normal(loc = 1, scale= 0.5),
              'Jwz_q1' : dists.Normal(loc = 0, scale= 0.5),
              'Jwx_q2' : dists.Normal(loc = -1, scale= 0.5),
              'Jwy_q2' : dists.Normal(loc = 0, scale= 0.5),
              'Jwz_q2' : dists.Normal(loc = 0, scale= 0.5),
              'kappa_x' : dists.TruncNormal(a= 0, b = 1e2, mu = 50, sigma=2),
              'kappa_y' : dists.TruncNormal(a= 1e2, b = 3e2, mu = 2e2, sigma=2)
            }
my_prior = dists.StructDist(prior_dict)
my_pmmh = mcmc.PMMH(ssm_cls=EndoscopeMotionModel_SSM, prior=my_prior, data=tip_rot_hyst, Nx=200,
                    niter=1000, verbose = True)
# %%
my_pmmh.run()
# %%

params = {}
for p in prior_dict.keys():  
    params[p] = my_pmmh.chain.theta[p][-1]

ssm_ = EndoscopeMotionModel_SSM(kappa_x = params["kappa_x"],
                                kappa_y = params["kappa_y"],
                                wq1 =  params["wq1"],
                                wq2 =  params["wq2"],
                                Jwx_q1 = params["Jwx_q1"],
                                Jwy_q1 = params["Jwy_q1"],
                                Jwz_q1 = params["Jwz_q1"],
                                Jwx_q2 = params["Jwx_q2"],
                                Jwy_q2 = params["Jwy_q2"],
                                Jwz_q2 = params["Jwz_q2"],
                                )

trajectory = ssm.simulate(T= len(q1))
rpy_traj_pmmh = np.zeros((len(q1),3))
for i in range (len(q1)):
    rpy_traj_pmmh[i]   = SO3.to_rpy(trajectory[0][i])
#%%
from particles import state_space_models as ssm
from particles.collectors import Moments
fk_model = ssm.Bootstrap(ssm=ssm_, data=tip_rot_hyst)  # we use the Bootstrap filter
pf = particles.SMC(fk=fk_model, N=100, collect=[Moments()], store_history=True)  # the algorithm
pf.run()  # actual computation
smooth_trajectories = pf.hist.extract_one_trajectory()
for i in range (len(q1)):
    rpy_traj_pmmh[i]   = SO3.to_rpy(smooth_trajectories[i])
# %%
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(rpy_sim[:,0], 'r', label = r'$roll_{sim}$')
ax1.plot(rpy_sim[:,1], 'g', label = r'$pitch_{sim}$')
ax1.plot(rpy_sim[:,2], 'b', label = r'$yaw_{sim}$')
ax1.plot(rpy_traj_pmmh[:,0], 'r--', label = r'$roll_{pmmh}$')
ax1.plot(rpy_traj_pmmh[:,1], 'g--', label = r'$pitch_{pmmh}$')
ax1.plot(rpy_traj_pmmh[:,2], 'b--', label = r'$yaw_{pmmh}$')
ax1.plot(rpy_traj[:,0], 'r:', label = r'$roll_{ssm}$')
ax1.plot(rpy_traj[:,1], 'g:', label = r'$pitch_{ssm}$')
ax1.plot(rpy_traj[:,2], 'b:', label = r'$yaw_{ssm}$')
ax1.grid()
ax1.set_ylabel('robot tip rot.\n(rad)')
ax1.set_xlabel('time(steps)')
ax1.legend(frameon = True, loc = 'center left', bbox_to_anchor= (1, 0.5))
plt.show()
# %%
import particles
from particles import smc_samplers as ssp

fk_smc2 = ssp.SMC2(ssm_cls=EndoscopeMotionModel_SSM, data=tip_rot_hyst, prior=my_prior,init_Nx=40,
                   ar_to_increase_Nx=0.1)
alg_smc2 = particles.SMC(fk=fk_smc2, N=40, verbose = True)
alg_smc2.run()
# %%
import seaborn as sb

# %%
params_smc2 = {}
idx = np.argmax(alg_smc2.X.lpost)
for p in prior_dict.keys():  
    params_smc2[p] = alg_smc2.X.theta[p][idx]

ssm = EndoscopeMotionModel_SSM(kappa_x = params_smc2["kappa_x"],
                                kappa_y = params_smc2["kappa_y"],
                                wq1 =  params_smc2["wq1"],
                                wq2 =  params_smc2["wq2"],
                                Jwx_q1 = params_smc2["Jwx_q1"],
                                Jwy_q1 = params_smc2["Jwy_q1"],
                                Jwz_q1 = params_smc2["Jwz_q1"],
                                Jwx_q2 = params_smc2["Jwx_q2"],
                                Jwy_q2 = params_smc2["Jwy_q2"],
                                Jwz_q2 = params_smc2["Jwz_q2"],
                                )
trajectory = ssm.simulate(T= len(q1))
rpy_traj_smc2 = np.zeros((len(q1),3))
for i in range (len(q1)):
    rpy_traj_smc2[i]   = SO3.to_rpy(trajectory[0][i])
# %%
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(rpy_sim[:,0], 'r', label = r'$roll_{sim}$')
ax1.plot(rpy_sim[:,1], 'g', label = r'$pitch_{sim}$')
ax1.plot(rpy_sim[:,2], 'b', label = r'$yaw_{sim}$')
ax1.plot(rpy_traj_pmmh[:,0], 'r--', label = r'$roll_{pmmh}$')
ax1.plot(rpy_traj_pmmh[:,1], 'g--', label = r'$pitch_{pmmh}$')
ax1.plot(rpy_traj_pmmh[:,2], 'b--', label = r'$yaw_{pmmh}$')
ax1.plot(rpy_traj_smc2[:,0], 'r:', label = r'$roll_{smc2}$')
ax1.plot(rpy_traj_smc2[:,1], 'g:', label = r'$pitch_{smc2}$')
ax1.plot(rpy_traj_smc2[:,2], 'b:', label = r'$yaw_{smc2}$')
ax1.grid()
ax1.set_ylabel('robot tip rot.\n(rad)')
ax1.set_xlabel('time(steps)')
ax1.legend(frameon = True, loc = 'center left', bbox_to_anchor= (1, 0.5))
plt.show()
# %%
