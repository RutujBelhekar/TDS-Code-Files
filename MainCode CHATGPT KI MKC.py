# ==========================================
# IMPORTS & BACKEND
# ==========================================
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# ==========================================
# CLASSICAL PARAMETERS (High-Resolution Setup)
# ==========================================
N = 100            # Particle Count
M = 1.0            # Mass per particle
dt = 0.00005       # MICRO-TIMESTEP: Prevents skipping past the potential wall
sub_steps = 100    # Keeps visualization fast (100 physics ticks per frame)
G = 0.2            # Gravitational constant
K_stiff = 2e-6     # Stiffness of the physical repulsion field
frames = 5000

# ==========================================
# INITIAL CONDITIONS
# ==========================================
initial_radius = 4.0
theta = np.random.uniform(0, 2 * np.pi, N)
r_dist = initial_radius * np.sqrt(np.random.rand(N))
pos = np.column_stack((r_dist * np.cos(theta), r_dist * np.sin(theta)))

# Tiny jitter to break perfect symmetry so they don't all hit (0,0) simultaneously
v = np.random.uniform(-0.05, 0.05, (N, 2))

# Precompute index pairs for vectorized O(N^2) calculations
i_idx, j_idx = np.tril_indices(N, k=-1)

# ==========================================
# PURE CONSERVATIVE PHYSICS ENGINE
# ==========================================
def get_forces_and_pe(pos):
    """Calculates perfectly conservative pure force fields"""
    diff = pos[i_idx] - pos[j_idx]
    # 1e-12 added only to prevent raw zero in initial np.sqrt
    dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
    
    # 1. Newtonian Gravity (Attractive)
    f_grav_mag = (G * M * M) / (dist**2)
    PE_grav = np.sum(- (G * M * M) / dist)
    
    # 2. Pure Hard-Core Repulsive Field (Inverse 12th power law)
    f_rep_mag = (12 * K_stiff) / (dist**13)
    PE_rep = np.sum(K_stiff / (dist**12))
    
    # 3. Net Force Assembly
    f_net_mag = f_rep_mag - f_grav_mag
    f_vec = (diff / dist[:, None]) * f_net_mag[:, None]
    
    f_net = np.zeros((N, 2))
    np.add.at(f_net, i_idx, f_vec)
    np.add.at(f_net, j_idx, -f_vec)
    
    Total_PE = PE_grav + PE_rep
    return f_net / M, Total_PE

# ==========================================
# INITIALIZATION & LIVE PLOTTING SETUP
# ==========================================
plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

scat = ax.scatter(pos[:, 0], pos[:, 1], s=35, c='cyan', edgecolors='none')
title = ax.set_title("Calculating Physics...", color='white')

# Get absolute initial energy for the Velocity Verlet setup
a, current_PE = get_forces_and_pe(pos)
initial_KE = np.sum(0.5 * M * np.sum(v**2, axis=1))
E_initial = current_PE + initial_KE

# ==========================================
# MAIN SIMULATION LOOP (Velocity Verlet)
# ==========================================
for step in range(frames):
    
    # Sub-stepping handles the microscopic dt
    for _ in range(sub_steps):
        # Velocity Verlet Integration
        v += 0.5 * a * dt          # 1. Half-step velocity
        pos += v * dt              # 2. Full-step position
        a, current_PE = get_forces_and_pe(pos) # 3. Re-evaluate forces precisely
        v += 0.5 * a * dt          # 4. Final half-step velocity
        
    # Visuals and Energy Tracking
    if step % 2 == 0:
        current_KE = np.sum(0.5 * M * np.sum(v**2, axis=1))
        current_E = current_KE + current_PE
        
        # Calculate the error (drift) from the absolute starting energy
        drift = current_E - E_initial
        
        scat.set_offsets(pos)
        title.set_text(f"Frame: {step*sub_steps} | Total E: {current_E:.4f} | Drift: {drift:.4f}")
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

plt.ioff()
plt.show()