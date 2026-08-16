import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ==========================
# GENTLE / STABLE PARAMETERS
# ==========================
N = 2200  # Particle count
R = 6.0  # Initial spawn radius
dt = 0.01  # Timestep

# Force strengths
k_att = 0.1  # Gentle pull
k_rep = 0.4  # Soft repulsion cushion
interaction_radius = 5.0  # Interaction zone
r_core = 0.5  # Core radius

# Center-of-mass internal damping
damping = 0.95  # Damps internal particle vibrations relative to COM

# ==========================
# SHAPE-DEFORMING PERTURBATION PARAMETERS
# ==========================
A = 70.5  # Shape deformation amplitude
omega = 1.0  # Driving frequency (rad/s)
time = 0.0  # Simulation time tracker

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


def compute_forces(x, y, t):
  """Computes internal cohesion + quadrupolar shape-deforming force."""
  dx = x[None, :] - x[:, None]
  dy = y[None, :] - y[:, None]
  d = np.sqrt(dx * dx + dy * dy)

  # Interaction mask
  mask = (d < interaction_radius) & (d > 0.01)

  # 1. Long-range attraction
  f_att = k_att * (interaction_radius - d)

  # 2. Short-range repulsion
  f_rep = k_rep * (r_core / np.maximum(d, 0.05)) ** 2

  # Net internal force matrix
  force_net = np.zeros_like(d)
  force_net[mask] = f_att[mask] - f_rep[mask]

  # Sum internal forces
  fx = np.sum(force_net * (dx / np.maximum(d, 1e-5)), axis=1)
  fy = np.sum(force_net * (dy / np.maximum(d, 1e-5)), axis=1)

  # 3. Quadrupolar Shape Deformation (X stretches, Y compresses)
  x_rel = x - np.mean(x)
  y_rel = y - np.mean(y)

  fx += A * x_rel * np.sin(omega * t)
  fy -= A * y_rel * np.sin(omega * t)

  return fx, fy


# Compute initial acceleration at t = 0
axx, ayy = compute_forces(x, y, time)

# ==========================
# FIGURE SETUP
# ==========================
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(x, y, s=15, c="dodgerblue", edgecolors="none", alpha=0.6)

ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title("Quadrupolar Shape Oscillations")


# ==========================
# LEAPFROG UPDATE LOOP
# ==========================
def update(frame):
  global x, y, vx, vy, axx, ayy, time

  # 1. Update Positions (Leapfrog step 1)
  x += vx * dt + 0.5 * axx * (dt**2)
  y += vy * dt + 0.5 * ayy * (dt**2)

  # Advance time
  time += dt

  # 2. Compute New Accelerations at t + dt
  new_axx, new_ayy = compute_forces(x, y, time)

  # 3. Update Velocities (Leapfrog step 2)
  vx += 0.5 * (axx + new_axx) * dt
  vy += 0.5 * (ayy + new_ayy) * dt

  # Update accelerations
  axx, ayy = new_axx, new_ayy

  # 4. Center-of-Mass Frame Damping
  v_cm_x = np.mean(vx)
  v_cm_y = np.mean(vy)

  vx = v_cm_x + (vx - v_cm_x) * damping
  vy = v_cm_y + (vy - v_cm_y) * damping

  sc.set_offsets(np.column_stack((x, y)))
  return (sc,)


ani = FuncAnimation(fig, update, frames=1000, interval=15, blit=True)
plt.show()