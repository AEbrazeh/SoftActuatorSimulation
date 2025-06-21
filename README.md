# A Soft Robotic Actuator Framework (Under Development)
A modular simulation framework for modeling soft robotic actuators using MuJoCo.  
This project focuses on segment-wise stiffness modulation to shape a soft structure in response to internal actuation. The framework includes automatic MuJoCo XML generation, geometry modeling via arc splines, and simulation tools for control and visualization.

---

## 🔧 Features

- Procedural generation of multi-segment soft actuators
- Stiffness modulation via rim-to-rim spring configuration
- XML generation for MuJoCo-compatible models
- Arc spline-based geometry (C¹-continuous chains of circular arcs)
- Tendon-based actuation
- Live simulation and rendering using MuJoCo
---

## 🧱 Project Structure

| `Actuator.py` | Main simulation class. Handles MuJoCo setup, stepping, visualization, and forces. |
| `xmlMake.py` | Programmatically builds the actuator model in MuJoCo XML format. |
| `arcSpline.py` | Defines the arc spline geometry used to model ideal actuator paths. |
| `Demonstration.ipynb` | Example notebook: builds an actuator, sets stiffness, simulates motion, and renders the result. |
| `README.md` | Project documentation. |

---

## 📐 Core Concept

This project explores how **stiffness modulation**—without direct control—can be used to shape soft structures. The actuator consists of rigid disks connected by springs, with actuation applied via tendons. By tuning the **stiffness pattern** along the structure, desired deformations can be achieved.

Target shapes are defined using arc splines, and future work will include optimization algorithms to automatically match actuator shape to arbitrary target curves.

---