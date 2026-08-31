# ===IMPORTS=== 

import numpy as np
import vispy
from vispy import app, scene
from vispy.scene.cameras import TurntableCamera
from vispy.scene.visuals import Markers, Text


# ===INITIAL CONDITIONS AND PARAMETERS===

dt=0.01 # Timestep (time interval between each frame)
M=1.0 # Mass of each particle
T_total=200 #total time of simulation
initial_radius=6.0 # Initial Spawn Radius
N=2000 # Particle Count
np.random.seed(42)
soft_core_parameter=0.35
current_time=0.0
steps_per_frame =1

u=np.random.uniform(-1,1,N) #Random Uniform Distribution for Z-axis

Theta=np.random.uniform(0,2*np.pi,N)
r=initial_radius*np.cbrt(np.random.rand(N))

#Particle Co-ordinates in Cartesian
x=r*np.sqrt(1-u**2)*np.cos(Theta) 
y=r*np.sqrt(1-u**2)*np.sin(Theta)
z=r*u
pos = np.column_stack((x, y, z))

#Velocity Components
v = np.zeros((N, 3))

#Acceleration Components
a = np.zeros((N, 3))

#Coeffients
G_grav = 1.2 # Co-efficient of Attractive Force (Directly Prop to Dist)
K_rep=0.4 # Co-efficient of Repulsive Force (Directly Prop to Dist)

# ===FUNCTION DEFINITIONS=== 

i_idx, j_idx = np.triu_indices(N, k=1)

def force_triangular(pos):
    # Use the global pre-calculated indices
    diff = pos[j_idx] - pos[i_idx]
    d = np.sqrt(np.sum(diff**2, axis=1) + soft_core_parameter**2)

    f_mag = (G_grav * M * M) / (d**2) - (K_rep / d**2)
    f_vec = (diff / d[:, None]) * f_mag[:, None]

    f_net = np.zeros_like(pos)
    np.add.at(f_net, i_idx, f_vec)
    np.add.at(f_net, j_idx, -f_vec)

    return f_net / M

# ====VisPy Setup====

canvas=scene.SceneCanvas(keys='interactive', size=(800,600), show=True, bgcolor='black', title='N-Body Particle Simulation')
view=canvas.central_widget.add_view()

view.camera=TurntableCamera(fov=45,distance=20)
markers=scene.visuals.Markers(parent=view.scene, scaling=False)
markers.set_data(pos=pos, face_color='white', size=1)

# ====Leapfrog Integration Loop====

a=force_triangular(pos)
v_half= v+0.5*dt*a

# ====Leapfrog Integration Loop====

a = force_triangular(pos)
v_half = v + 0.5 * dt * a

def update(event):
    global pos, v_half, current_time

    # ALL of this must be indented 4 spaces so it belongs to the function
    for _ in range(steps_per_frame):    
        if current_time >= T_total:
            print(f"Simulation complete! Reached T={T_total}")
            timer.stop()
            return  # Tells the function to completely stop running

        pos += v_half * dt
        a_new = force_triangular(pos)
        v_half += a_new * dt
        current_time += dt

    # These also belong inside the function, aligned with the 'for' loop
    markers.set_data(pos=pos.copy())
    canvas.update()

timer = app.Timer(interval=0, connect=update, start=True)
if __name__ == '__main__':
    app.run()