# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Mahdi Chaari (chaari.mahdi@outlook.com)

"""
Created on 24/05/2024
"""
import numpy as np
import matplotlib.pyplot as plt

from backlash_simulation_examples.models.backlash.generic_backlash import AbstractBacklash


def sinkhole_function(x_in: np.ndarray, w: float) -> np.ndarray:
    '''
    The sinkhole function addresses the lack of motion transmission
    when the proximal actuator changes the direction of its motion
    f(x) = 1 - exp(-x**2/w)

    Parameters
    ----------
        x_in :  ndarray
            the displacement of the actuator in a single direction
            (clockwise or counter clockwise)
        w : float
            the width of the backlash
    
    Returns
    --------
        output of the sinkhole function
    '''
    return 1 - np.exp(-(x_in**2) / w)

class BacklashSinkhole(AbstractBacklash):
    """
    Continuous Backlash Model based on the sinkhole function
    whose parameters are the (constant) backlash width and the slope of the characteristic.
    
        Overview
        ------
            x_out(k) = x_out(k-1) + dq(k) * m * sinkhole_function(x_in, w)
            
            with dq(k) = q(k) - q(k-1) ==> the change in the actuator input q

            and x_in an **internal state** that keeps track of the displacement 
                    of the actuator in a single direction (clockwise or counter clockwise)
    """
    def __init__(self,
                 params : dict,
                 q_label : str = 'undefined', q_unit : str = 'undefined',
                 c_label :str = 'undefined', c_unit : str = 'undefined') :
        # Invoke parent constructor (generic task model)
        super().__init__(parameters_count = 2,
                         parameters_labels = ["w", "m"], # width, slope
                         parameters_units =  [q_unit, c_unit+"/"+q_unit],
                         q_label = q_label, q_unit = q_unit,
                         c_label= c_label, c_unit = c_unit
        )

        # Update module name
        self._model_name = "2 dof continuous backlash with constant width and slope based on the sinkhole function"

        # set the model parameters
        self.w = params['w'] # width of backlash
        self.m = params['m'] # slope of backlash

        # initialize the internal state
        self.x_in = 0

    def evaluate(self, q_k : np.ndarray, q_k_1  : np.ndarray, x_k_1 : np.ndarray = None, ) -> np.ndarray:
        '''
        Evaluate the internal state of the backlash element and returns True if successful.
        This function should be called at every time step before retrieving the state transition function
        Once called, the state transition function f can be retrieved.

        Parameters
        ----------
        x_k_1_ : np.array
            previous state.
        q_k : float
            current actuator position.
        q_k_1 : float, optional
            Previous  actuator position.

        Returns
        -------
        f : Callable
            the state transition function
        '''
        dq = q_k - q_k_1 # the change in actuator input
        # update the internal state
        self.x_in +=  dq 
        # check if the actuator has changed direction
        if (np.sign(self.x_in) != 0) and (np.sign(self.x_in) != np.sign(dq)) :
            self.x_in = 0
        
        return self.f(q_k, q_k_1)

    def f(self, x_k_1, q_k) -> np.ndarray :
        raise NotImplemented
    
    def f(self, q_k, q_k_1 ) -> np.ndarray :
        '''
        State transition function
        f = \Delta q_{distal}
        Returns
        -------
        f : Callable
            Actuator displacement transmitted to the distal side between timesteps k-1 and k
        '''
        _f = lambda q_k, q_k_1 : (q_k-q_k_1) * self.m * sinkhole_function(self.x_in, self.w)
        return _f
    
    def equivalent_backlash_width(self, x_k : np.ndarray) -> float:
        return self.w

    def equivalent_mean_slope(self, x_k : np.ndarray) -> float:
        return self.m

    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> float:
        return 0.0

def sinkhole_backlash_model (q : np.ndarray, params : dict) -> np.ndarray:
    '''
    Computes the output of the sinkhole backlash model
    x_out(k) = x_out(k-1) + dq(k) * m * sinkhole_function(x_in, w)
    with dq(k) = q(k) - q(k-1) ==> the change in the actuator input q
    and  x_in the displacement of the actuator in a single direction 
            (clockwise or counter clockwise)

    Parameters
    ----------
        q : ndarray
            the actuator displacement
        params : dict
            the parameters of the model
            m : input gain 
            w : backlash width
    
    Returns
    -------
        x_out : ndarray
            the output of the sinkhole model

    '''
    dq = np.zeros(q.shape)
    dq[1:] = q[1:] - q[:-1]

    x_in = np.zeros(q.shape)
    x_out = np.zeros(q.shape)
    for k in range(1,len(q)):
        x_in[k] = x_in[k - 1] + dq[k]
        if (np.sign(x_in[k]) != 0) and (np.sign(x_in[k]) != np.sign(dq[k])) :
            x_in[k] = 0
        
        x_out[k] = x_out[k-1] + dq[k] * params['m'] * sinkhole_function(x_in[k], params['w'])

    return(x_out)

if __name__ == "__main__" :

    params = [{'w': 0.1 , 'm': 1}, {'w': 0.3 , 'm': 1}, {'w': 0.5 , 'm': 1}]
    t_eval = np.linspace(0, 40, 40000)
    fig = plt.figure()
    for param in params :
        q = 0.5* np.sin(0.5  * t_eval)
        x_out = sinkhole_backlash_model(q = q, params= param)

        # SB_q = BacklashSinkhole(
        #             params= params, q_label= "q", q_unit= "rad",
        #                             c_label= "c", c_unit= "rad")

        # x_out_class = np.zeros(t_eval.shape)
        # for i in range(1,len(t_eval)):
        #     f = SB_q.evaluate(q[i], q[i-1])
        #     x_out_class[i] = x_out_class[i-1] + f(q[i], q[i-1])


        '''
        Plotting results
        '''
        
        # plt.plot(t_eval, q, label = 'x_in')
        # plt.plot(t_eval, x_out, label = 'x_out')
        # plt.plot(t_eval, q_, label = 'x_in_')
        # plt.plot(t_eval, x_out_, label = 'x_out_')
        
        # #plt.plot(t_eval, x_out_class, label = 'x_out_class')
        # plt.legend()
        # plt.title("sinkhole backlash model output w.r.t time")
        # plt.xlabel("time")
        # plt.show()

        #plt.plot(q,x_out, label ='func')
        

        plt.plot(q,x_out)
    #plt.plot(q,x_out_class, label = 'cls')
    plt.title("Sinkhole Backlash model")
    plt.xlabel('Proximal Cable Displacement (rad)')
    plt.ylabel('Distal  Cable  Displacement (rad)')
    plt.legend(["w=0.1", "w=0.3", "w=0.5"])
    plt.show()
    fig.savefig('sinkhole backlash model_w.svg')

    t= np.linspace(-5, 5, 40000)
    fig = plt.figure()
    plt.plot(t, sinkhole_function(t,0.1))
    plt.plot(t, sinkhole_function(t,0.3))
    plt.plot(t, sinkhole_function(t,0.5))
    plt.legend(["w=0.1", "w=0.3", "w=0.5"])
    plt.title("Inverted Gaussian Function")
    fig.savefig('inverted_gaussian_w.svg')
    plt.show()