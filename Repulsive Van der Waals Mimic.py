import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ==========================
# GENTLE / STABLE PARAMETERS
# ==========================
N = 2000  # Particle count
R = 6.0  # Initial spawn radius
dt = 0.01  # Smaller timestep for smoother math

# Significantly reduced force strengths
k_att = 0.1  # Gentle pull 
k_rep = 0.4  # Soft repulsion cushion 
interaction_radius = 5.0  # Wider, smoother interaction zone
r_core = 0.5  # Smaller core radius

# Stronger damping to absorb collision shockwaves
damping = 0.97  # Removes 3% of excess velocity each frame

# ==========================
# INITIAL PARTICLES
# ==========================
np.random.seed(42)
theta = np.random.uniform(0, 2 * np.pi, N)
r = R * np.sqrt(np.random.rand(N))

x = r * np.cos(theta)
y = r * np.sin(theta)

vx = np.zeros(N)
vy = np.zeros(N)

# Initialize accelerations for Leapfrog
axx = np.zeros(N)
ayy = np.zeros(N)

# ==========================
# FORCE FUNCTION
# ==========================

def compute_forces(x, y):
    """Computes pairwise attractive + short-range repulsive forces."""
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    d = np.sqrt(dx * dx + dy * dy)

    # Interaction mask: ignore self-interaction (d > 0.01) and far particles
    mask = (d < interaction_radius) & (d > 0.01)

    # 1. Long-range attraction
    f_att = k_att * (interaction_radius - d)

    # 2. Short-range repulsion (Van der Waals-like core)
    f_rep = k_rep * (r_core / np.maximum(d, 0.05)) ** 2

    # Net force: positive = attraction, negative = repulsion
    force_net = np.zeros_like(d)
    force_net[mask] = f_att[mask] - f_rep[mask]

    # Sum forces along direction vectors
    fx = np.sum(force_net * (dx / np.maximum(d, 1e-5)), axis=1)
    fy = np.sum(force_net * (dy / np.maximum(d, 1e-5)), axis=1)

    return fx, fy

# Compute initial acceleration
axx, ayy = compute_forces(x, y)

# ==========================
# FIGURE SETUP
# ==========================
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(x, y, s=13, c="dodgerblue", edgecolors="none", alpha=0.6)

ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title("Moon Particles with Van der Waals")

# ==========================
# LEAPFROG UPDATE LOOP
# ==========================
def update(frame):
    global x, y, vx, vy, axx, ayy

    # 1. Update Positions (Leapfrog step 1)
    x += vx * dt + 0.5 * axx * (dt**2)
    y += vy * dt + 0.5 * ayy * (dt**2)

    # 2. Compute New Accelerations
    new_axx, new_ayy = compute_forces(x, y)

    # 3. Update Velocities (Leapfrog step 2)
    vx += 0.5 * (axx + new_axx) * dt
    vy += 0.5 * (ayy + new_ayy) * dt

    # Update current accelerations
    axx, ayy = new_axx, new_ayy

    # 4. Apply gentle relative damping
    vx *= damping
    vy *= damping

    sc.set_offsets(np.column_stack((x, y)))
    return (sc,)

ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)
plt.show()