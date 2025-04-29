import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mediapy as media
from xmlMake import *

class softActuator:
    def __init__(self, timeStep, nSegments, nDisks, length, radius, mass, innerStiffness, innerDamping, outerStiffness, outerDamping, gear, yieldStrain, hardeningRatio, fillRatio=0.5):
        createSegment('Actuator-{}'.format(nSegments), nSegments, length, radius, nDisks, innerStiffness, innerDamping, 4, gear, mass, timeStep, fillRatio)
        with open('Actuator-{}.xml'.format(nSegments), "r") as f:
            self.xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.N = nDisks
        
        self.outerStiffness = outerStiffness * (nDisks - 1) * nSegments
        self.outerDamping = outerDamping * (nDisks - 1) * nSegments
        
        self.yieldStrain = yieldStrain
        
        self.nOuterSprings = 4 * (nDisks-1) * nSegments
        self.alpha = hardeningRatio
        self.yS = np.ones(self.nOuterSprings) * self.model.tendon_lengthspring[:self.nOuterSprings, 0] * self.yieldStrain
        self.p = np.zeros(self.nOuterSprings)
        
        self.stiffness = np.ones(self.nOuterSprings) * self.outerStiffness
        self.damping = np.ones(self.nOuterSprings) * self.outerStiffness
        
    def updateStiffness(self):
        v = self.data.ten_velocity[:self.nOuterSprings]
        dl = (self.data.ten_length - self.model.tendon_lengthspring[:, 0])[:self.nOuterSprings]
        e = (dl - self.p).clip(None, self.yS)
        dp = (dl - self.p - e).clip(0, None)
        self.p += dp
        F = self.stiffness * e + self.stiffness * self.alpha * self.p #+ self.damping * v
        self.data.ctrl[:self.nOuterSprings] = -F
        
    def setStiffness(self, segment, spring, percentage):
        start = (self.N-1) * (4*segment + spring)
        end = (self.N-1) * (4*segment + spring + 1)
        self.stiffness[start:end] = percentage * self.outerStiffness
        #self.damping[start:end] = percentage * self.outerDamping


    def step(self, ctrlInput):
        self.data.ctrl[-1] = ctrlInput
        self.updateStiffness()
        mujoco.mj_step(self.model, self.data)
        
    def reset(self):
        self.yS = np.ones(self.nOuterSprings) * self.model.tendon_lengthspring[:self.nOuterSprings, 0] * self.yieldStrain
        self.p = np.zeros(self.nOuterSprings)
        
        self.stiffness = np.ones(self.nOuterSprings) * self.outerStiffness
        self.damping = np.ones(self.nOuterSprings) * self.outerStiffness
        
        self.data.ctrl = 0
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)