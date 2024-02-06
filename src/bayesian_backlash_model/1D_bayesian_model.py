import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
import matplotlib.animation as animation

sign = lambda x: x and (1, -1)[x<0]

# Generate time values
time = np.linspace(0, 1000, 200000)
pos_des = np.piecewise(time, [time < 250, ((time < 500) & (time >= 250)), ((time < 750) & (time > 500)), time>=750], [lambda time: np.sin(1/4*time) ,lambda time: 2*np.sin(1/4*time),lambda time: 4*np.sin(1/4*time),lambda time: 6*np.sin(1/4*time)])
#pos_des = np.sin(time)

b_mean = 0.6
b_cov = 0.3
x_n = []
x_n_1 = 0
u_n_1 = 0
pos_n_1 = 0
q_one_way = 0
b_noise = norm.rvs(loc=0, scale=b_cov)
for pos in pos_des:
    b = b_mean + b_noise
    u_n = pos - pos_n_1
    if u_n * u_n_1 < 0 :
        b_noise = norm.rvs(loc=0, scale=b_cov)
        q_one_way = 0
        b_left = b
    else :
        b_left = max(b - abs(q_one_way), 0)
        q_one_way += u_n
        
    x_noise = norm.rvs(loc=0, scale=abs(0.005))
    #noise = 0
    # update the state
    x_n.append(x_n_1 + sign(u_n)*max(abs(u_n)-b_left,0)+x_noise) 
    b_left = max(b_left - abs(u_n), 0)
    
    x_n_1 = x_n[-1]
    u_n_1 = u_n
    pos_n_1 = pos



plt.plot(time, pos_des)
plt.plot(time, x_n)
plt.show()

plt.plot(pos_des,x_n)
plt.show()