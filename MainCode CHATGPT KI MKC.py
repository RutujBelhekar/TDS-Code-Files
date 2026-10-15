# ===IMPORTS=== 
import numpy as np
import vispy
from vispy import app, scene
from vispy.scene.cameras import PanZoomCamera
from vispy.scene.visuals import Markers

# ===INITIAL CONDITIONS AND PARAMETERS===
dt = 0.005               
M = 1.0                  
T_total = 600            
initial_radius = 4.6       
N = 500                  # 500 ensures smooth live performance for O(N^2) repulsion
np.random.seed(42)
soft_core_parameter = 0.35
current_time = 0.0

steps_per_frame = 4      

# 2D Circular Spawn (using sqrt for uniform flat distribution)
Theta = np.random.uniform(0, 2*np.pi, N)
r = initial_radius * np.sqrt(np.random.rand(N))

# Particle Co-ordinates in 2D (X, Y)
x = r * np.cos(Theta) 
y = r * np.sin(Theta)
pos = np.column_stack((x, y))

# ZERO INITIAL VELOCITY (Direct radial collapse in 2D)
v = np.zeros((N, 2))

# Coefficients
G_grav = 0.25 
K_rep = 0.01   

# ===FUNCTION DEFINITIONS=== 

# Pre-calculate pairs for O(N^2) interparticle repulsion
i_idx, j_idx = np.triu_indices(N, k=1)

def force_hybrid(pos):
    # ==========================================
    # 1. GRAVITY (Pull to Center of Mass) - O(N)
    # ==========================================
    com = np.mean(pos, axis=0)
    diff_com = com - pos # Vector pointing TOWARD the center of mass
    d_com = np.sqrt(np.sum(diff_com**2, axis=1) + soft_core_parameter**2)

    # Gravity scaled by N*M to simulate the collective mass of the cluster
    f_mag_grav = (G_grav * M * (N * M)) / (d_com**2)
    f_grav = (diff_com / d_com[:, None]) * f_mag_grav[:, None]

    # ==========================================
    # 2. REPULSION (Interparticle Stacking) - O(N^2)
    # ==========================================
    diff_part = pos[j_idx] - pos[i_idx]
    
    # Softcore applied to interparticle distance
    d_part = np.sqrt(np.sum(diff_part**2, axis=1) + soft_core_parameter**2)

    # Hard-wall repulsion (d^4) so particles physically bounce off each other
    f_mag_rep = K_rep / (d_part**4)
    f_vec_rep = (diff_part / d_part[:, None]) * f_mag_rep[:, None]

    f_rep = np.zeros_like(pos)
    np.add.at(f_rep, i_idx, -f_vec_rep) # Push i away from j
    np.add.at(f_rep, j_idx, f_vec_rep)  # Push j away from i

    # ==========================================
    # 3. COMBINE FORCES
    # ==========================================
    f_net = f_grav + f_rep
    return f_net / M

# ====VisPy Setup====
canvas = scene.SceneCanvas(keys='interactive', size=(1024, 768), show=True, bgcolor='black', title='2D Hybrid Simulation')
view = canvas.central_widget.add_view()

# Switched to a 2D Pan/Zoom camera
view.camera = PanZoomCamera(aspect=1)
view.camera.rect = (-10, -10, 20, 20) # Initial zoom bounds

markers = scene.visuals.Markers(parent=view.scene, scaling=False)
# INCREASED SIZE HERE:
markers.set_data(pos=pos, face_color='white', size=8)

# ====Leapfrog Integration Loop====
a = force_hybrid(pos)
v_half = v + 0.5 * dt * a

def update(event):
    global pos, v_half, current_time

    for _ in range(steps_per_frame):    
        if current_time >= T_total:
            print(f"Simulation complete! Reached T={T_total}")
            timer.stop()
            return  

        pos += v_half * dt
        a_new = force_hybrid(pos)
        v_half += a_new * dt
        current_time += dt

    # Keep camera tracking the center of mass
    com = np.mean(pos, axis=0)
    view.camera.center =tuple(com)

    # Push to GPU directly
    markers.set_data(pos=pos.copy())
    canvas.update()

# Start the live timer immediately
timer = app.Timer(interval=0, connect=update, start=True)

if __name__ == '__main__':
    app.run()