import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ==========================
# SIMULATION PARAMETERS
# ==========================
N = 1500
R = 10.0   
dt = 0.005
time = 0.0 # Can be a bit larger now that we fixed the explosion

# Newtonian Gravity
G = 2.5    # Increased slightly to hold the cloud together
M = 1.0    

# Pseudo-Van der Waals (Softened for computer stability)
k_vdw = 15.0   # Re-balanced for the new power law
r_core = 0.5   

# Gentle Spin
v_spin = 3.5   # Lowered! If this is too high, they escape gravity and scatter
damping = 0.99 # Dust friction to help them settle into a disk

# ==========================
# INITIAL PARTICLES
# ==========================
np.random.seed(42)
theta = np.random.uniform(0, 2 * np.pi, N)
r = R * np.sqrt(np.random.rand(N))

x = r * np.cos(theta)
y = r * np.sin(theta)

# Gentle Tangential Spin
vx = -v_spin * np.sin(theta)
vy =  v_spin * np.cos(theta)

axx = np.zeros(N)
ayy = np.zeros(N)

# ==========================
# FORCE FUNCTION
# ==========================
# Add time 't' to the function arguments
def compute_forces(x, y, t):
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    d = np.sqrt(dx * dx + dy * dy)

    mask = d > 0.01
    d_safe = np.maximum(d, 0.2)

    # 1. NEWTONIAN GRAVITY
    f_grav = np.zeros_like(d)
    f_grav[mask] = (G * (M ** 2)) / (d_safe[mask] ** 2)

    # 2. SOFTENED REPULSION
    f_vdw = np.zeros_like(d)
    f_vdw[mask] = k_vdw * (r_core / d_safe[mask]) ** 4

    force_net = np.zeros_like(d)
    force_net[mask] = f_grav[mask] - f_vdw[mask]

    fx = np.sum(force_net * (dx / d_safe), axis=1)
    fy = np.sum(force_net * (dy / d_safe), axis=1)

    # ====================================================
    # 3. EXTERNAL TIDAL PERTURBATION (The new addition)
    # ====================================================
    A = 0.5      # Perturbation strength
    omega = 1.0  # Perturbation frequency
    
    x_rel = x - np.mean(x)
    y_rel = y - np.mean(y)
    
    # This will stretch and squeeze the spinning disk periodically
    fx += A * x_rel * np.sin(omega * t)
    fy -= A * y_rel * np.sin(omega * t)

    return fx, fy

axx, ayy = compute_forces(x, y, time)

# ==========================
# FIGURE SETUP
# ==========================
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(x, y, s=8, c="darkorange", edgecolors="none", alpha=0.6)

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.set_title("Stable Accretion Disk", color='white')

# ==========================
# LEAPFROG UPDATE LOOP
# ==========================
def update(frame):
    global x, y, vx, vy, axx, ayy, time

    x += vx * dt + 0.5 * axx * (dt**2)
    y += vy * dt + 0.5 * ayy * (dt**2)

    new_axx, new_ayy = compute_forces(x, y, time)

    vx += 0.5 * (axx + new_axx) * dt
    vy += 0.5 * (ayy + new_ayy) * dt
    axx, ayy = new_axx, new_ayy

    v_cm_x = np.mean(vx)
    v_cm_y = np.mean(vy)
    vx = v_cm_x + (vx - v_cm_x) * damping
    vy = v_cm_y + (vy - v_cm_y) * damping
    time += dt
    
    # Speed limit kept just in case of rare multi-particle collisions
    speed = np.sqrt(vx**2 + vy**2)
    max_speed = 40.0
    overspeed_mask = speed > max_speed
    if np.any(overspeed_mask):
        vx[overspeed_mask] = (vx[overspeed_mask] / speed[overspeed_mask]) * max_speed
        vy[overspeed_mask] = (vy[overspeed_mask] / speed[overspeed_mask]) * max_speed

    sc.set_offsets(np.column_stack((x, y)))
    return (sc,)

ani = FuncAnimation(fig, update, frames=1500, interval=15, blit=True)
plt.show()