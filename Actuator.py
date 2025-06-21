import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
from xmlMake import *

def length(x, r, theta, phi, psi):
    return np.sqrt(-2 * r**2 * np.sin(phi) * np.sin(psi) * np.sin(theta) * np.cos(psi)
                   +2 * r**2 * np.sin(psi)**2 * np.cos(phi)
                   -2 * r**2 * np.sin(psi)**2 * np.cos(theta)
                   -2 * r**2 * np.cos(phi)
                   +2 * r**2
                   -2 * r * x * np.sin(phi) * np.cos(psi)
                   +2 * r * x * np.sin(psi) * np.sin(theta) * np.cos(phi)
                   +x**2)
    
def pose2angles(position, orientation, N):
    # Step 1: Get rotation vector
    rotvec = R.from_matrix(orientation).as_rotvec()
    beta = np.linalg.norm(rotvec)
    
    # Handle edge case: no rotation (flat)
    if beta < 1e-9:
        return np.array([0.0, 0.0, np.linalg.norm(position) / (N-1)])
    
    n = rotvec / beta
    
    # Step 2: Extract bending direction in YZ plane
    ny = n[1]
    nz = n[2]
    alpha = np.arctan2(ny, nz)
    
    # Step 3: Radius
    radius = np.linalg.norm(position) / (2 * np.sin(beta / 2))
    
    # Step 4: Arc parameters
    delta = radius * beta / (N - 1)
    theta = beta * np.sin(alpha) / (N - 1)
    phi = beta * np.cos(alpha) / (N - 1)

    return np.array([theta, phi, delta])

    
def calculateStiffness(Position1, Orientation1, x0, x01, x02, x03, k, F, N, r):
    
    theta, phi, x = pose2angles(Position1, Orientation1, N)
        
    l1 = length(x, r, theta, phi, 0)
    l2 = length(x, r, theta, phi, 2 * np.pi / 3)
    l3 = length(x, r, theta, phi, 4 * np.pi / 3)
    
    A = np.array([
        [
            -(N-1) * (l1 - x01) * (r * np.sin(phi) - x) / l1,
            +(N-1) * (l2 - x02) * (r * np.sin(phi) + np.sqrt(3) * r * np.sin(theta) * np.cos(phi) + 2 * x) / (2 * l2),
            +(N-1) * (l3 - x03) * (r * np.sin(phi) - np.sqrt(3) * r * np.sin(theta) * np.cos(phi) + 2 * x) / (2 * l3)
        ],
        [
            0,
            +(N-1) * r * (l2 - x02) * (np.sqrt(3) * r * np.sin(phi) * np.cos(theta) + 3 * r * np.sin(theta) + 2 * np.sqrt(3) * x * np.cos(phi) * np.cos(theta)) / (4 * l2),
            -(N-1) * r * (l3 - x03) * (np.sqrt(3) * r * np.sin(phi) * np.cos(theta) - 3 * r * np.sin(theta) + 2 * np.sqrt(3) * x * np.cos(phi) * np.cos(theta)) / (4 * l3)
        ],
        [
            +(N-1) * r * (l1 - x01) * (r * np.sin(phi) - x * np.cos(phi)) / l1,
            +(N-1) * r * (l2 - x02) * (r * np.sin(phi) + np.sqrt(3) * r * np.sin(theta) * np.cos(phi) - 2 * np.sqrt(3) * x * np.sin(phi) * np.sin(theta) + 2 * x * np.cos(phi)) / (4 * l2),
            +(N-1) * r * (l3 - x03) * (r * np.sin(phi) - np.sqrt(3) * r * np.sin(theta) * np.cos(phi) + 2 * np.sqrt(3) * x * np.sin(phi) * np.sin(theta) + 2 * x * np.cos(phi)) / (4 * l3)
        ]
    ])
    
    b = np.array([
        F
        - 3 * (N-1) * k * x
        + ((N-1) * k * r * x0 * np.sin(phi)) / (2 * l3)
        - (np.sqrt(3) * (N-1) * k * r * x0 * np.sin(theta) * np.cos(phi)) / (2 * l3)
        + ((N-1) * k * x * x0) / l3
        + ((N-1) * k * r * x0 * np.sin(phi)) / (2 * l2)
        + (np.sqrt(3) * (N-1) * k * r * x0 * np.sin(theta) * np.cos(phi)) / (2 * l2)
        + ((N-1) * k * x * x0) / l2
        - ((N-1) * k * r * x0 * np.sin(phi)) / l1
        + ((N-1) * k * x * x0) / l1,

        - (3 * (N-1) * k * r**2 * np.sin(theta)) / 2
        - (np.sqrt(3) * (N-1) * k * r**2 * x0 * np.sin(phi) * np.cos(theta)) / (4 * l3)
        + (3 * (N-1) * k * r**2 * x0 * np.sin(theta)) / (4 * l3)
        - (np.sqrt(3) * (N-1) * k * r * x * x0 * np.cos(phi) * np.cos(theta)) / (2 * l3)
        + (np.sqrt(3) * (N-1) * k * r**2 * x0 * np.sin(phi) * np.cos(theta)) / (4 * l2)
        + (3 * (N-1) * k * r**2 * x0 * np.sin(theta)) / (4 * l2)
        + (np.sqrt(3) * (N-1) * k * r * x * x0 * np.cos(phi) * np.cos(theta)) / (2 * l2),

        - (3 * (N-1) * k * r**2 * np.sin(phi)) / 2
        + ((N-1) * k * r**2 * x0 * np.sin(phi)) / (4 * l3)
        - (np.sqrt(3) * (N-1) * k * r**2 * x0 * np.sin(theta) * np.cos(phi)) / (4 * l3)
        + (np.sqrt(3) * (N-1) * k * r * x * x0 * np.sin(phi) * np.sin(theta)) / (2 * l3)
        + ((N-1) * k * r * x * x0 * np.cos(phi)) / (2 * l3)
        + ((N-1) * k * r**2 * x0 * np.sin(phi)) / (4 * l2)
        + (np.sqrt(3) * (N-1) * k * r**2 * x0 * np.sin(theta) * np.cos(phi)) / (4 * l2)
        - (np.sqrt(3) * (N-1) * k * r * x * x0 * np.sin(phi) * np.sin(theta)) / (2 * l2)
        + ((N-1) * k * r * x * x0 * np.cos(phi)) / (2 * l2)
        + ((N-1) * k * r**2 * x0 * np.sin(phi)) / l1
        - ((N-1) * k * r * x * x0 * np.cos(phi)) / l1
    ])
    
    return np.linalg.solve(A, b)

class softActuator:
    def __init__(self, timeStep, nSegments, nDisks, length, radius, mass, innerStiffness, innerDamping, outerStiffness, outerDamping, gear, fillRatio=0.5):
        createSegment('Actuator-{}'.format(nSegments), nSegments, length, radius, nDisks, innerStiffness, innerDamping, outerStiffness, outerDamping, 3, gear, mass, timeStep, fillRatio)
        with open('Actuator-{}.xml'.format(nSegments), "r") as f:
            self.xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.M = nSegments
        self.N = nDisks
        
        self.outerStiffness = outerStiffness * (self.N - 1)
        self.outerDamping = outerDamping * (self.N - 1)
        
    def setStiffness(self, segment, spring, percentage):
        stiffness = np.full(self.N - 1, self.outerStiffness * percentage)
        damping = np.full(self.N - 1, self.outerDamping * percentage)
        self.model.tendon_stiffness[(3 * (self.M + segment) + spring) * (self.N - 1):(3 * (self.M + segment) + spring + 1) * (self.N - 1)] = stiffness
        self.model.tendon_damping[(3 * (self.M + segment) + spring) * (self.N - 1):(3 * (self.M + segment) + spring + 1) * (self.N - 1)] = damping

    def step(self, ctrlInput):
        self.data.ctrl = ctrlInput
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        self.data.ctrl = 0
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
