# ===== IMPORTS ======

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ===== MOON PARAMETERS ======

N = 2200  # Particle Count
R = 6.0  # Initial Spawn Radius
dt = 0.01  # Timestep

# ===== FORCE PARAMETERS =====

K_att = 0.1  # Co-efficient of Attractive Force (Directly Prop to Dist)
K_rep = 0.4  # Co-efficient of Repulsive Force (Directly Prop to R-core)
Interaction_radius = 5.0  # Limit of Attractive Force w other Particles
r_core = 0.5  # Core Radius of Repulsive Force (Directly Prop to Dist)

# ===== DAMPING PARAMETERS =====

damping = 0.95  # Damps internal vibrations of particles from COM.

# ===== PERTURBATION PARAMETERS =====

A = 70.5  # Amplitude
Omega = 6.0  # Frequency (rad/s)
time = 0.0  # Simulation Time Tracker
perturbation_on = True  # Toggle state for external perturbation

# ===== FORMATION LOGIC FOR SPAWNING PARTICLES =====

np.random.seed(42)

Theta = np.random.uniform(0, 2 * np.pi, N)  # Random Angle Generation for Particles
r = R * np.sqrt(np.random.rand(N))  # Random Radius Generation for Particles

# Conversion to Cartesian from Polar Co-ords

x = r * np.cos(Theta)
y = r * np.sin(Theta)

# Parts initial Zero Velocity Components to count in leapfrog

vx = np.zeros(N)
vy = np.zeros(N)

# Initialize accelerations for Leapfrog

axx = np.zeros(N)
ayy = np.zeros(N)


# ===== FORCE FUNCTION =====


def compute_forces(x, y, t):
  """Computes internal cohesion + toggleable quadrupolar shape-deforming force."""
  dx = x[None, :] - x[:, None]
  dy = y[None, :] - y[:, None]
  d = np.sqrt(dx * dx + dy * dy)

  # Interaction mask
  mask = (d < Interaction_radius) & (d > 0.01)

  # 1. Long-range attraction
  f_att = K_att * (Interaction_radius - d)

  # 2. Short-range repulsion
  f_rep = K_rep * (r_core / np.maximum(d, 0.05)) ** 2

  # Net internal force matrix
  force_net = np.zeros_like(d)
  force_net[mask] = f_att[mask] - f_rep[mask]

  # Sum internal forces
  fx = np.sum(force_net * (dx / np.maximum(d, 1e-5)), axis=1)
  fy = np.sum(force_net * (dy / np.maximum(d, 1e-5)), axis=1)

  # 3. Quadrupolar Shape Deformation (Only applied when toggled ON)
  if perturbation_on:
    x_rel = x - np.mean(x)
    y_rel = y - np.mean(y)

    fx += A * x_rel * np.sin(Omega * t)
    fy -= A * y_rel * np.sin(Omega * t)

  return fx, fy


# Compute initial acceleration at t = 0

axx, ayy = compute_forces(x, y, time)

# ==========================
# DARK THEME FIGURE SETUP
# ==========================
fig, ax = plt.subplots(figsize=(7, 7.5))
plt.subplots_adjust(bottom=0.16)  # Reserve space at the bottom for the button

fig.patch.set_facecolor("black")
ax.set_facecolor("black")

# White particles keeping exact same size, edgecolors, and alpha
sc = ax.scatter(x, y, s=15, c="white", edgecolors="none", alpha=0.6)

ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_aspect("equal")
ax.grid(True, color="white", alpha=0.15)
ax.set_title("Quadrupolar Shape Oscillations", color="white", pad=12)

# Style ticks and border spines for the black background
ax.tick_params(colors="white")
for spine in ax.spines.values():
  spine.set_edgecolor("white")

# ==========================
# ON / OFF TOGGLE BUTTON
# ==========================
ax_button = plt.axes([0.36, 0.04, 0.28, 0.055])
btn_toggle = Button(
    ax_button, "Perturbation: ON", color="#1b4d3e", hovercolor="#286e59"
)
btn_toggle.label.set_color("white")
btn_toggle.label.set_fontsize(10)
btn_toggle.label.set_fontweight("bold")
for spine in ax_button.spines.values():
  spine.set_edgecolor("white")


def toggle_perturbation(event):
  global perturbation_on, time
  perturbation_on = not perturbation_on

  if perturbation_on:
    time = 0.0  # Reset phase so sin(Omega * t) ramps up smoothly from 0
    btn_toggle.label.set_text("Perturbation: ON")
    btn_toggle.color = "#1b4d3e"
    btn_toggle.hovercolor = "#286e59"
  else:
    btn_toggle.label.set_text("Perturbation: OFF")
    btn_toggle.color = "#2b2b2b"
    btn_toggle.hovercolor = "#444444"

  ax_button.set_facecolor(btn_toggle.color)
  fig.canvas.draw_idle()


btn_toggle.on_clicked(toggle_perturbation)


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