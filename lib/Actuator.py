import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

class softActuator:
    def __init__(self, config):
        self.dt = config['timeStep']
        self.M = config['numSegments']
        self.N = config['numDisks']
        self.l = config['length']
        self.r = config['radius']
        self.m = config['mass']
        self.iK = config['innerStiffness'] * (self.N - 1)
        self.iD = config['innerDamping'] * (self.N - 1)

        self.oK = config['outerStiffness'] * (self.N - 1)
        self.oD = config['outerDamping'] * (self.N - 1)

        self.gear = config['gear']
        
        self.x0 = np.full((4,), self.l / ((self.N-1) * self.M))
        
        self.lowerBound = np.array([self.l, 0, 0])
        self.upperBound = np.array([4*self.l, 2*np.pi/3, 2*np.pi])
        
        with open('{}'.format(config['file']), "r") as f:
            self.xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
    
    def setStiffness(self, curve, F):
        '''
        Calculate the required stiffness of a single actuator based on the curve parameters and initial conditions.
        '''
        if curve.shape != (self.M, 3):
            raise ValueError("Curve must be of shape (M, 3) where M is the number of segments.")
        
        oK = []
        
        for ii, curve_ in enumerate(curve):
            length, beta, alpha = curve_
            
            l0 = length / (self.N - 1)
            theta = -beta * np.sin(alpha) / (self.N - 1)
            phi = beta * np.cos(alpha) / (self.N - 1)
            
            cPhi   = np.cos(phi)
            sPhi   = np.sin(phi)
            cTheta = np.cos(theta)
            sTheta = np.sin(theta)
            sqrt3  = np.sqrt(3)

            delta = np.sqrt(l0**2 + self.x0[0]**2 * cTheta**2 * cPhi**2 - self.x0[0]**2) - self.x0[0] * cTheta * cPhi


            psi = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
            l1, l2, l3 = np.sqrt(-2 * self.r**2 * sPhi * np.sin(psi) * sTheta * np.cos(psi)
                                +2 * self.r**2 * np.sin(psi)**2 * cPhi
                                -2 * self.r**2 * np.sin(psi)**2 * cTheta
                                -2 * self.r**2 * cPhi
                                +2 * self.r**2
                                -2 * self.r * delta * sPhi * np.cos(psi)
                                +2 * self.r * delta * np.sin(psi) * sTheta * cPhi
                                -2 * self.r * self.x0[0] * sPhi * np.cos(psi) * cTheta
                                +2 * self.r * self.x0[0] * np.sin(psi) * sTheta
                                +2 * delta * self.x0[0] * cPhi * cTheta
                                +delta**2 + self.x0[0]**2)
                        
            A = np.array([
                [ (self.N-1) * (l1 - self.x0[1]) * (-self.r*sPhi + delta + self.x0[0]*cPhi*cTheta) / l1,
                (self.N-1) * (l2 - self.x0[2]) * (self.r*sPhi + sqrt3*self.r*sTheta*cPhi + 2*delta + 2*self.x0[0]*cPhi*cTheta) / (2*l2),
                (self.N-1) * (l3 - self.x0[3]) * (self.r*sPhi - sqrt3*self.r*sTheta*cPhi + 2*delta + 2*self.x0[0]*cPhi*cTheta) / (2*l3)],

                [ (self.N-1) * (l1 - self.x0[1]) * (self.r*sPhi - delta*cPhi) * sTheta * self.x0[0] / l1,
                (self.N-1) * (l2 - self.x0[2]) * (sqrt3*self.r**2*sPhi*cTheta + 3*self.r**2*sTheta + 2*sqrt3*self.r*delta*cPhi*cTheta - 2*self.r*self.x0[0]*sPhi*sTheta + 2*sqrt3*self.r*self.x0[0]*cTheta - 4*delta*self.x0[0]*sTheta*cPhi ) / (4*l2),
                -(self.N-1) * (l3 - self.x0[3]) * (sqrt3*self.r**2*sPhi*cTheta - 3*self.r**2*sTheta + 2*sqrt3*self.r*delta*cPhi*cTheta + 2*self.r*self.x0[0]*sPhi*sTheta + 2*sqrt3*self.r*self.x0[0]*cTheta + 4*delta*self.x0[0]*sTheta*cPhi) / (4*l3)],

                [-(self.N-1) * (l1 - self.x0[1]) * (-self.r**2*sPhi + self.r*delta*cPhi + self.r*self.x0[0]*cPhi*cTheta + delta*self.x0[0]*sPhi*cTheta ) / l1,
                (self.N-1) * (l2 - self.x0[2]) * (self.r**2*sPhi + sqrt3*self.r**2*sTheta*cPhi - 2*sqrt3*self.r*delta*sPhi*sTheta + 2*self.r*delta*cPhi + 2*self.r*self.x0[0]*cPhi*cTheta - 4*delta*self.x0[0]*sPhi*cTheta) / (4*l2),
                (self.N-1) * (l3 - self.x0[3]) * (self.r**2*sPhi - sqrt3*self.r**2*sTheta*cPhi + 2*sqrt3*self.r*delta*sPhi*sTheta + 2*self.r*delta*cPhi + 2*self.r*self.x0[0]*cPhi*cTheta - 4*delta*self.x0[0]*sPhi*cTheta) / (4*l3)]
            ])
            
            b = np.array([
                F * self.gear * (delta + self.x0[0] * cPhi * cTheta) / l0
                -3 * self.iK * (delta + self.x0[0] * cPhi * cTheta)
                +self.iK * self.x0[0] * (-self.r * sPhi + delta + self.x0[0] * cPhi * cTheta) / l1
                +self.iK * self.x0[0] * (self.r * sPhi + sqrt3 * self.r * sTheta * cPhi + 2 * delta + 2 * self.x0[0] * cPhi * cTheta) / (2 * l2)
                +self.iK * self.x0[0] * (self.r * sPhi - sqrt3 * self.r * sTheta * cPhi + 2 * delta + 2 * self.x0[0] * cPhi * cTheta) / (2 * l3),
                
                -F * self.gear * delta * self.x0[0] * sTheta * cPhi / l0
                -1.5 * self.iK * self.r**2 * sTheta
                +3 * self.iK * delta * self.x0[0] * sTheta * cPhi
                +self.iK * self.x0[0]**2 * (self.r * sPhi - delta * cPhi) * sTheta / l1
                +self.iK * self.x0[0] * ( sqrt3 * self.r**2 * sPhi * cTheta + 3 * self.r**2 * sTheta + 2 * sqrt3 * self.r * delta * cPhi * cTheta - 2 * self.r * self.x0[0] * sPhi * sTheta + 2 * sqrt3 * self.r * self.x0[0] * cTheta - 4 * delta * self.x0[0] * sTheta * cPhi) / (4 * l2)
                +self.iK * self.x0[0] * (-sqrt3 * self.r**2 * sPhi * cTheta + 3 * self.r**2 * sTheta - 2 * sqrt3 * self.r * delta * cPhi * cTheta - 2 * self.r * self.x0[0] * sPhi * sTheta - 2 * sqrt3 * self.r * self.x0[0] * cTheta - 4 * delta * self.x0[0] * sTheta * cPhi) / (4 * l3),

                -F * self.gear * delta * self.x0[0] * sPhi * cTheta / l0
                -1.5 * self.iK * self.r**2 * sPhi
                +3 * self.iK * delta * self.x0[0] * sPhi * cTheta
                +(self.iK * self.x0[0] * (self.r**2 * sPhi- self.r * delta * cPhi - self.r * self.x0[0] * cPhi * cTheta - delta * self.x0[0] * sPhi * cTheta)) / l1
                +(self.iK * self.x0[0] * (self.r**2 * sPhi + sqrt3 * self.r**2 * sTheta * cPhi - 2 * sqrt3 * self.r * delta * sPhi * sTheta + 2 * self.r * delta * cPhi + 2 * self.r * self.x0[0] * cPhi * cTheta - 4 * delta * self.x0[0] * sPhi * cTheta)) / (4 * l2)
                +(self.iK * self.x0[0] * (self.r**2 * sPhi - sqrt3 * self.r**2 * sTheta * cPhi + 2 * sqrt3 * self.r * delta * sPhi * sTheta + 2 * self.r * delta * cPhi + 2 * self.r * self.x0[0] * cPhi * cTheta - 4 * delta * self.x0[0] * sPhi * cTheta)) / (4 * l3)
                ])
            
            ok_ = np.linalg.solve(A, b)
            if (ok_ < 0).any():
                raise ValueError("Calculated stiffness values cannot be negative. Please check the input parameters for segment {}.".format(ii))
            ok_ = np.repeat(ok_, self.N - 1)
            oK.append(ok_)
        oK = np.array(oK).flatten()
        self.model.tendon_stiffness[3*self.M*(self.N-1):6*self.M*(self.N-1)] = oK * (self.N - 1)
    
    
    def step(self, ctrlInput):
        self.data.ctrl = ctrlInput
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        self.data.ctrl = 0
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
