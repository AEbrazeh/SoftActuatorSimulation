import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mediapy as media
from xmlMake import *
from Actuator import *
import time

height = width = 1024
dt = 0.01

# Input force stats:
H = 1
s = 0
T = 10
alpha = 0

# Create the actuator model
actuator = softActuator(nSegments= 2,
                         length = 0.15,
                         radius = 0.03,
                         nDisks = 10,
                         innerStiffness = 100,
                         innerDamping = 20,
                         outerStiffness= 300,
                         outerDamping = 0,
                         gear = 80,
                         mass = 0.1,
                         yieldStrain= 0.5,
                         timeStep = dt,
                         hardeningRatio=0,
                         fillRatio=0.1)

# A single demonstration of the actuator with a force profile of a ramp up and down
duration = 10 # seconds
counter = 0
actuator.reset()
with mujoco.viewer.launch_passive(actuator.model, actuator.data) as viewer:
    actuator.setStiffness(0, 2, 0.5)
    actuator.setStiffness(1, 0, 0.5)    
    while viewer.is_running():
        step_start = time.time()
        actuator.step(np.clip(2 * H * min(counter*dt - s, s + T - counter*dt) / ((1 - alpha) * T), 0, H) )
        viewer.sync()
        time_until_next_step = actuator.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        counter += 1
        
duration = 20 # seconds  
# checking the effect of parameters on the actuator
F = np.zeros(int(duration/dt))
t = np.zeros(int(duration/dt))
l = np.zeros((11, 11, int(duration/dt)))

for jj in range(11):
    for ii in range(11):
        actuator = softActuator(nSegments= 2,
                                length = 0.15,
                                radius = 0.03,
                                nDisks = 10,
                                innerStiffness = 100,
                                innerDamping = 20,
                                outerStiffness = 300,
                                outerDamping = 0,
                                gear = 80,
                                mass = 0.1,
                                yieldStrain = 0.1 + 0.02 * ii,
                                timeStep = dt,
                                hardeningRatio = jj/10,
                                fillRatio = 0.1)
        actuator.reset()
        actuator.setStiffness(0, 2, 0.5)
        actuator.setStiffness(1, 0, 0.5)
        counter = 0
        while np.round(actuator.data.time, 1-int(np.log10(dt))) <= duration:
            F[counter] = np.clip(2 * H * min(counter*dt - s, s + T - counter*dt) / ((1 - alpha) * T), 0, H)
            t[counter] = actuator.data.time
            l[jj, ii, counter] = actuator.data.ten_length[-1]
            actuator.step(F[counter])
            counter += 1

plt.title('Effect of hardening ratio')
plt.plot(l[:, 0].T, F)
plt.show()
