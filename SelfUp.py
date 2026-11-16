import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ===== SYSTEM & FORCE PARAMETERS =====
m = 1.0
n = 200
inirad = 4.2
softcore = 0.35
G = 4.0
Krep = 0.005
dt = 0.001          # Increased from 0.0001 so live animation moves at visible speed

# ===== FRICTION (DAMPING) & PERTURBATION PARAMETERS =====
Kdamp = 0.5         # Viscous drag coefficient (bleeds off collapse energy)
A = 15.0            # Perturbation amplitude (scaled to match G=4.0, n=200)
Omega = 2.0         # Perturbation angular frequency (rad/s)
time = 0.0
perturbation_on = False  # Starts OFF at the beginning

np.random.seed(67)

r = inirad * np.sqrt(np.random.rand(n))
theta = np.random.uniform(0, 2 * np.pi, n)
x = r * np.cos(theta)
y = r * np.sin(theta)

pos = np.column_stack((x, y))
v = np.zeros((n, 2))

i_idx, j_idx = np.tril_indices(n, k=-1)


def forces(pos, v, t):
    diff = pos[i_idx] - pos[j_idx]
    dsq = (diff**2).sum(axis=1) + softcore**2
    d = np.sqrt(dsq)

    # 1. Internal Gravity & Repulsion
    Fgrav = G * m * m / d**2
    Frep = Krep / d**12
    Fnet = Frep - Fgrav

    Fvec = (diff / d[:, None]) * Fnet[:, None]
    Ftot = np.zeros((n, 2))
    np.add.at(Ftot, i_idx, Fvec)
    np.add.at(Ftot, j_idx, -Fvec)

    # 2. Viscous Friction (Center-of-Mass frame so cluster doesn't freeze in drift)
    v_cm = np.mean(v, axis=0)
    F_drag = -Kdamp * (v - v_cm)
    Ftot += F_drag

    # 3. Toggleable Quadrupolar Perturbation (X stretches, Y compresses)
    if perturbation_on:
        pos_rel = pos - np.mean(pos, axis=0)
        wave = A * np.sin(Omega * t)
        Ftot[:, 0] += wave * pos_rel[:, 0]
        Ftot[:, 1] -= wave * pos_rel[:, 1]

    anet = Ftot / m
    return anet, Ftot


# Initial acceleration at t = 0
anet, Fnet = forces(pos, v, time)

# ===== FIGURE & UI SETUP =====
fig, ax = plt.subplots(figsize=(8, 8.5))
plt.subplots_adjust(bottom=0.16)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title("Self-Gravitating Cluster with Toggleable Perturbation")

scat = ax.scatter(pos[:, 0], pos[:, 1], s=25, alpha=0.8)

# ===== ON / OFF TOGGLE BUTTON (STARTS OFF) =====
ax_button = plt.axes([0.35, 0.04, 0.30, 0.055])
btn_toggle = Button(
    ax_button, "Perturbation: OFF", color="#d9d9d9", hovercolor="#bfbfbf"
)
btn_toggle.label.set_fontsize(10)
btn_toggle.label.set_fontweight("bold")


def toggle_perturbation(event):
    global perturbation_on, time
    perturbation_on = not perturbation_on

    if perturbation_on:
        time = 0.0  # Reset phase so sin(Omega * t) ramps smoothly from 0
        btn_toggle.label.set_text("Perturbation: ON")
        btn_toggle.color = "#4caf50"
        btn_toggle.hovercolor = "#45a049"
    else:
        btn_toggle.label.set_text("Perturbation: OFF")
        btn_toggle.color = "#d9d9d9"
        btn_toggle.hovercolor = "#bfbfbf"

    ax_button.set_facecolor(btn_toggle.color)
    fig.canvas.draw_idle()


btn_toggle.on_clicked(toggle_perturbation)

# ===== LIVE VELOCITY VERLET LOOP =====
steps_per_frame = 15  # Sub-steps per rendered frame for smooth, fast motion


def update(frame):
    global pos, v, anet, Fnet, time

    for _ in range(steps_per_frame):
        v += anet * 0.5 * dt
        pos += v * dt
        time += dt
        anet, Fnet = forces(pos, v, time)
        v += anet * 0.5 * dt

    scat.set_offsets(pos)
    return (scat,)


ani = FuncAnimation(fig, update, frames=2000, interval=15, blit=True)
plt.show()