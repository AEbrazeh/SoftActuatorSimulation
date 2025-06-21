# A Soft Robotic Actuator Framework
---
A modular simulation framework for modeling soft robotic actuators using MuJoCo.  
This project focuses on segment-wise stiffness modulation to shape a soft structure in response to internal actuation. The framework includes automatic MuJoCo XML generation, geometry modeling via arc splines, and simulation tools for control and visualization.
**⚠️ Note:** This project is currently under active development. Features, APIs, and structure may change.
---

## Overview

This repository contains a complete, parametric workflow for creating, simulating, and analysing multi-segment soft robotic actuators.  The system couples
1. **Geometry generation** – continuous arc splines that describe the desired neutral shape;
2. **Automatic model synthesis** – procedural generation of MuJoCo XML files with bodies, springs, tendons, and actuators; and
3. **Physics-based simulation** – a Python wrapper that steps, visualises, and probes the resulting model in real time.

---

## Key Features

- **Segment-wise stiffness control** (rim-to-rim linear springs elements)
- **Arc-spline geometry** with \(C^1\) continuity
- **Fully procedural MuJoCo XML generation**
- **Headless or on-screen rendering**

---

## Repository Structure

| Path                  | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `Actuator.py`         | High-level simulation wrapper (load, reset, step, render).  |
| `arcSpline.py`        | Arc-segment and spline utilities (geometry backbone).       |
| `xmlMake.py`          | Functions that build the MuJoCo XML model programmatically. |
| `Demonstration.ipynb` | End-to-end notebook: generate model → simulate → visualise. |
| `README.md`           | Project documentation (this file).                          |

---
