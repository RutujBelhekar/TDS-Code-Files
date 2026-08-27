# ===IMPORTS=== 

import numpy as np
import vispy
from vispy import app, scene
from vispy.scene.cameras import TurntableCamera
from vispy.scene.visuals import Markers, Text


# ===INITIAL CONDITIONS AND PARAMETERS===

dt=0.01 # Timestep (time interval between each frame)
M=1.0 # Mass of each particle
T_total=2000 #total time of simulation
initial_radius=6.0 # Initial Spawn Radiusde
N=200 # Particle Count
np.random.seed(42)

u=np.random.uniform(-1,1,N) #Random Uniform Distribution for Z-axis

Theta=np.random.uniform(0,2*np.pi,N)
Theata_2=np.random.uniform(0,2*np.pi,N)
r=initial_radius*np.cbrt(np.random.rand(N))

#Particle Co-ordinates in Cartesian
x=r*np.sqrt(1-u**2)*np.cos(Theta) 
y=r*np.sqrt(1-u**2)*np.sin(Theta)
z=r*u
pos = np.column_stack((x, y, z))

#Velocity Components
v = np.zeros((N, 3))

#Coeffients
K_att=1.0 # Co-efficient of Attractive Force (Directly Prop to Dist)
K_rep=0.4 # Co-efficient of Repulsive Force (Directly Prop to Dist)

# ===FUNCTION DEFINITIONS=== 
