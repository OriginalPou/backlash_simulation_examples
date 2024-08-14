# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 12/08/2024
"""

from mpi4py import MPI

import os
import numpy as np
import scipy.io

from models.endoscope_motion_model import EndoscopeMotionModel_SSM

from particles import distributions as dists
from particles import multiSMC, SMC
from particles import state_space_models as ssm
from particles import smc_samplers as ssp

# load the trajectory of actuator 1 and 2 (in rad)
path = os.path.dirname(os.path.abspath(__file__))
q1 = scipy.io.loadmat(path + '/data/csi/q1.mat')['q1']
q2 = scipy.io.loadmat(path + '/data/csi/q2.mat')['q2']

# load the evolution of the rotation of the tip when the bending part is considered (with Backlash)
tip_rot_hyst = scipy.io.loadmat(path + '/data/csi/tip_rot_hyst.mat')['tip_rot_hyst']
tip_rot_hyst = np.transpose(tip_rot_hyst, (2, 0, 1))
    

EndoscopeMotionModel_SSM.Set_Actuation(q1=q1, q2=q2)

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

prior_dict = {'wq1': dists.TruncNormal(mu= 0.5, sigma = 0.5, a = 0, b = 2),
              'wq2': dists.TruncNormal(mu= 0.5, sigma = 0.5, a = 0, b = 2),
              'Jwx_q1' : dists.Normal(loc = -0.24, scale= 0.5),
              'Jwy_q1' : dists.Normal(loc = 0.95, scale= 0.5),
              'Jwz_q1' : dists.Normal(loc = 0.11, scale= 0.5),
              'Jwx_q2' : dists.Normal(loc = -0.78, scale= 0.5),
              'Jwy_q2' : dists.Normal(loc = -0.01, scale= 0.5),
              'Jwz_q2' : dists.Normal(loc = 0.1, scale= 0.5),
            }

my_prior = dists.StructDist(prior_dict)

fk_smc2 = ssp.SMC2(ssm_cls=EndoscopeMotionModel_SSM,fk_cls = ssm.GuidedPF, \
                    data=tip_rot_hyst, prior=my_prior, init_Nx=50, ar_to_increase_Nx=0.1,
                    mpi = 1)

if rank == 0 :
    alg_smc2 = SMC(fk=fk_smc2, N=size * 8, verbose = True)
    alg_smc2.run()
    print(alg_smc2.cpu_time)
    
else:
    request = None
    while True :
        request = comm.bcast(request, root=0)
        # print(request)
        fk_smc2.run_pf_mpi(request['N'], request['Nx'], request['t'], request['move'])
        
        try:
            if fk_smc2.x_pfs[0].t >= len(fk_smc2.data) :
                break
        except:
            pass
