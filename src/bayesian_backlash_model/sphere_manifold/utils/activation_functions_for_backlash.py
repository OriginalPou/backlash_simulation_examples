import numpy as np
import matplotlib.pyplot as plt

from sympy import tanh, plot, lambdify
from sympy.abc import x


# plot((tanh(x-2) + tanh(-x - 2))/2 +1 , (x, -10, 10))
# plot((tanh(x/2-2) + tanh(-x/2 - 2))/2 +1 , (x, -10, 10))

time = np.linspace(0, 160 * 2 * np.pi, 200000)
act_pos = np.piecewise(time, [time < 40 * 2 * np.pi, ((time < 80 * 2 * np.pi) & (time >= 40 * 2 * np.pi)),\
                               ((time < 120 * 2 * np.pi) & (time >= 80 * 2 * np.pi)), time>=120 * 2 * np.pi],\
                                        [lambda time: np.sin(1/4*time) ,lambda time: 2*np.sin(1/4*time), \
                                        lambda time: 4*np.sin(1/4*time),lambda time: 6*np.sin(1/4*time)])
act_dpos = np.copy(act_pos)
act_dpos[1:] -= act_pos[:-1]

# check how much we've pulled in the same direction
act_dir = [act_dpos[0]]
for i in range(len(act_pos[1:])):
    if (np.sign(act_dir[i-1]) == 0 or np.sign(act_dir[i-1]) == np.sign(act_dpos[i])):
        act_dir.append( act_dir[i-1] + act_dpos[i])
    else : 
        act_dir.append(0)
    

backlash = (tanh(x-2) + tanh(-x - 2))/2 +1
backlash_num = lambdify(x, backlash, "numpy")

dist_pos = []
dist_pos.append(act_dpos[0] * backlash_num(act_pos[0]))

for i in range(len(act_pos[1:])):
    dist_pos.append(dist_pos[i-1] + act_dpos[i] * backlash_num(act_dir[i]))

plt.plot (time, act_pos, label = "actuator pos")
plt.plot (time, dist_pos, label = "distal pos")
plt.show()

plt.plot(act_pos, dist_pos, label = "b_w = 1, slope = 1")

dist_pos = []
dist_pos.append(act_dpos[0] * backlash_num(act_pos[0]/2))
for i in range(len(act_pos[1:])):
    dist_pos.append(dist_pos[i-1] + act_dpos[i] * backlash_num(act_dir[i]/2))

plt.plot(act_pos, dist_pos, label = "b_w = 2, slope = 1")


dist_pos = []
dist_pos.append(2* act_dpos[0] * backlash_num(act_pos[0]))
for i in range(len(act_pos[1:])):
    dist_pos.append(dist_pos[i-1] + 2* act_dpos[i] * backlash_num(act_dir[i]))

plt.plot(act_pos, dist_pos, label = "b_w = 1, slope = 2")


dist_pos = []
dist_pos.append(2*act_dpos[0] * backlash_num(act_pos[0]/2))
for i in range(len(act_pos[1:])):
    dist_pos.append(dist_pos[i-1] + 2* act_dpos[i] * backlash_num(act_dir[i]/2))

plt.plot(act_pos, dist_pos, label = "b_w = 2, slope = 2")
plt.xlabel("actuator position")
plt.ylabel("distal position")
plt.title("Hysteresis using tanh activation function")
plt.legend()
plt.show()
