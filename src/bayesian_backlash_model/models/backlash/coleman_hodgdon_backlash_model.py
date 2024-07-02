# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 22/05/2024
"""
#%%
from sympy import lambdify, symbols
from sympy import sin, Abs
from sympy.abc import t
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

'''
REF: Nonlinear friction modelling and compensation control of hysteresis phenomena for a pair of tendon-sheath actuated surgical robots, 
     Author: T.N.Do et al.
     DOI: 10.1016/j.ymssp.2015.01.001
'''

def coleman_hodgdon_model (params : dict,
                           x_in : symbols):
    '''
    Defines the Coleman Hodgdon Hysteresis model
    x_out = c * x_in + x_h
    x_h_p = beta * x_in_p - alpha * abs(x_in_p) * x_h
    
    Parameters
    ----------
        params : dict 
            the parameters c, alpha, beta
        x_in : symbols
            the displacement at the proximal end
            and its derivative w.r.t time

    Returns
    --------
        x_h : symbols
            the hysteresis function of the coleman_hodgdon_model
        x_h_p : symbols
            the time derivative of x_h 
    '''
    x_h = symbols('x_h')
    x_h_p = symbols('x_h_p')

    x_h_p = params['beta'] * x_in.diff(t) - params['alpha'] * Abs(x_in.diff(t)) * x_h
    return(x_h, x_h_p)

def coleman_hodgdon_model_euler (params : dict,
                                    x_in : np.ndarray,
                                    t_eval : np.ndarray) -> np.ndarray:
    '''
    Computes the output of the Coleman Hodgdon Hysteresis model using the Euler method
    x_out(k) = c * x_in(k) + x_h(k)
    
    Discretizing the ODE
    \Delta(x_h)(k) = beta * \Delta(x_in)(k) - alpha * abs(\Delta(x_in)(k)) * x_h(k-1)
    x_h(0) = 0
    x_h(k) = x_h(k-1) + \Delta(x_h)(k)
    
    Parameters
    ----------
        params : dict 
            the parameters c, alpha, beta
        x_in : ndarray
            the displacement at the proximal end
        t_eval : ndarray
            the timesteps for evaluating the discrete ODE

    Returns
    --------
        x_out : ndarray
            the output of the Coleman Hodgdon model            
    '''
    d_x_in = np.zeros(t_eval.shape)
    d_x_h  = np.zeros(t_eval.shape)

    x_out  = np.zeros(t_eval.shape)
    x_h    = np.zeros(t_eval.shape)
    
    for k in range(1,len(t_eval)):
        d_x_in[k] =  x_in[k] - x_in[k-1]
        d_x_h[k]  =  params['beta'] * d_x_in[k] - params['alpha'] * np.abs(d_x_in[k]) * d_x_h[k-1]
        x_h[k]    =  x_h[k-1] + d_x_h[k]
        x_out[k]  =  params['c'] * x_in[k] + x_h[k] 

    return(x_out)

#%%
if __name__ == "__main__" :
    
    '''
    RK45 implementation
    '''
    motor_shaft_radius = 0.02
    x_in = 0.02 * t * sin(0.5 * t) # in rad
    params = {'c': 1 , 'beta': -0.91 , 'alpha': 10.1}
    x_h, x_h_p = coleman_hodgdon_model(params = params, x_in= x_in)

    # evaluate numerically
    t_eval = np.linspace(0, 40, 40000)
    t_span = (0,40)

    f = lambdify((t,x_h), x_h_p)
    x_h_num = scipy.integrate.solve_ivp(f, t_span, [0], t_eval=t_eval).y

    x_in_lambda = lambdify(t, x_in)
    x_in_num = x_in_lambda(t_eval)

    x_out_num = params['c'] * x_in_num + x_h_num # in rad


    '''
    Plotting results
    '''
    plt.plot(t_eval, x_in_num, label = 'x_in')
    plt.plot(t_eval, x_out_num[0], label = 'x_out')
    plt.legend()
    plt.title("coleman hodgdon model output w.r.t time")
    plt.xlabel("time")
    plt.show()

    fig = plt.figure()
    plt.plot(x_in_num,x_out_num[0], label = 'EK45')
    plt.title("Coleman-Hodgdon model")
    plt.xlabel('Proximal Cable Displacement (rad)')
    plt.ylabel('Distal  Cable  Displacement (rad)')
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig('Coleman-Hodgdon.svg')

    #%%
    '''
    Euler implementation
    NOTE The Euler implementation is fairly dissimilar to the RK45 implementation
    '''
    x_in_dis = 0.02 * t_eval * np.sin(1/4 * t_eval)
    x_out_dis = coleman_hodgdon_model_euler(params= params, x_in=x_in_dis, t_eval= t_eval)

    '''
    Plotting results
    '''
    plt.plot(t_eval, x_in_num, label = 'x_in')
    plt.plot(t_eval, x_in_dis, label = 'x_in(discrete)')
    plt.plot(t_eval, x_out_num[0], label = 'x_out')
    plt.plot(t_eval, x_out_dis, label = 'x_out(discrete)')
    plt.legend()
    plt.title("coleman hodgdon model output w.r.t time")
    plt.xlabel("time")
    plt.show()

    plt.plot(x_in_num,x_out_num[0], label = 'EK45')
    plt.plot(x_in_dis,x_out_dis, label = 'Euler')
    plt.title("distal displacement w.r.t proximal displacement")
    plt.xlabel("x_in")
    plt.ylabel("x_out")
    plt.legend()
    plt.show()







# %%
