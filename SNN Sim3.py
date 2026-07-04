import nengo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# IEEE CSL QUALITY FIGURE SETTINGS
# =====================================================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["font.size"] = 14
rcParams["axes.titlesize"] = 15
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["legend.fontsize"] = 12
rcParams["lines.linewidth"] = 1.8
rcParams["axes.linewidth"] = 1.1
rcParams["savefig.dpi"] = 600

# =====================================================
# Hybrid SNN Chaos Control of Lorenz System
# =====================================================

dt = 0.001
sim_time = 25.0

sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

K = 75*np.array([10.0, 10.0, 10.0])

plant_state = np.array([5.0, 5.0, 25.0])

# -----------------------------------------------------
# Desired Circular Orbit
# -----------------------------------------------------
def target_orbit(t):
    R = 1.0
    w = 0.7
    return [
        R*np.cos(w*t),
        R*np.sin(w*t),
        1.0
    ]

# -----------------------------------------------------
# Controlled Lorenz Plant
# -----------------------------------------------------
def lorenz_plant(t, u):
    global plant_state

    x, y, z = plant_state
    ux, uy, uz = u

    dx = sigma*(y-x) + ux
    dy = x*(rho-z) - y + uy
    dz = x*y - beta*z + uz

    plant_state += dt*np.array([dx, dy, dz])

    return plant_state

# -----------------------------------------------------
# Features [x,y,z,ex,ey,ez]
# -----------------------------------------------------
def state_error_func(x):
    s = x[:3]
    d = x[3:]
    e = d - s
    return np.concatenate([s, e])

# -----------------------------------------------------
# Nonlinear SNN Compensation
# -----------------------------------------------------
def snn_dynamics(v):
    x, y, z, ex, ey, ez = v

    u1 = 0.7*np.tanh(y) + 0.15*z*ex
    u2 = 0.5*np.sin(x) + 0.12*x*ez
    u3 = 0.4*np.tanh(x-y) + 0.1*y*ey

    return [u1, u2, u3]

# -----------------------------------------------------
# Linear feedback
# -----------------------------------------------------
def feedback_func(v):
    s = v[:3]
    d = v[3:]
    e = d - s
    return K*e

# =====================================================
# Build Nengo Model
# =====================================================
model = nengo.Network(label="Hybrid Chaos Control")

with model:

    plant = nengo.Node(lorenz_plant, size_in=3, size_out=3)
    ref   = nengo.Node(target_orbit)

    combo = nengo.Ensemble(
        n_neurons=3000,
        dimensions=6,
        radius=5
    )

    nengo.Connection(plant, combo[:3], synapse=None)
    nengo.Connection(ref, combo[3:], synapse=None)

    features = nengo.Ensemble(
        n_neurons=500,
        dimensions=6,
        radius=5
    )

    nengo.Connection(combo, features,
                     function=state_error_func,
                     synapse=0.1)

    snn_out = nengo.Node(size_in=3)

    nengo.Connection(features, snn_out,
                     function=snn_dynamics,
                     synapse=0.1)

    fb = nengo.Node(size_in=3)

    nengo.Connection(combo, fb,
                     function=feedback_func,
                     synapse=0.05)

    control = nengo.Node(size_in=3)

    nengo.Connection(snn_out, control, synapse=None)
    nengo.Connection(fb, control, synapse=None)

    nengo.Connection(control, plant, synapse=None)

    p_state = nengo.Probe(plant, synapse=0.1)
    p_ref   = nengo.Probe(ref, synapse=0.1)
    p_u     = nengo.Probe(control, synapse=0.1)

# =====================================================
# Run Simulation
# =====================================================
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(sim_time)

X = sim.data[p_state]
R = sim.data[p_ref]

# =====================================================
# Uncontrolled Lorenz
# =====================================================
steps = int(sim_time/dt)

chaos = np.zeros((steps,3))
xc = np.array([5.0,5.0,25.0])

for i in range(steps):

    x,y,z = xc

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    xc += dt*np.array([dx,dy,dz])
    chaos[i] = xc

# =====================================================
# Error
# =====================================================
err = np.linalg.norm(X-R, axis=1)
time = np.arange(len(err))*dt

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# IEEE CSL STYLE (Times + Larger Fonts)
# =====================================================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"

rcParams["font.size"] = 16
rcParams["axes.titlesize"] = 17
rcParams["axes.labelsize"] = 15
rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13
rcParams["legend.fontsize"] = 13
rcParams["axes.linewidth"] = 1.2
rcParams["savefig.dpi"] = 700

# =====================================================
# ONLY SUBFIGURES (1) AND (2)
# assumes chaos, X, R already computed
# =====================================================

fig = plt.figure(figsize=(14, 6))

# -----------------------------------------------------
# SUBFIG 1 : Uncontrolled Lorenz Attractor
# -----------------------------------------------------
ax1 = fig.add_subplot(121, projection='3d')

ax1.plot(
    chaos[:,0], chaos[:,1], chaos[:,2],
    color='blue',
    lw=0.85
)

ax1.view_init(elev=24, azim=-58)

ax1.set_title("(a) Uncontrolled Lorenz Attractor", pad=16)
ax1.set_xlabel("x", labelpad=8)
ax1.set_ylabel("y", labelpad=8)
ax1.set_zlabel("z", labelpad=8)

# cleaner IEEE look
ax1.grid(False)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# -----------------------------------------------------
# SUBFIG 2 : Controlled Trajectory
# -----------------------------------------------------
ax2 = fig.add_subplot(122, projection='3d')

# controlled state
ax2.plot(
    X[:,0], X[:,1], X[:,2],
    color='tab:blue',
    lw=1.8,
    label="Controlled"
)

# desired orbit
ax2.plot(
    R[:,0], R[:,1], R[:,2],
    '--',
    color='red',
    lw=1.6,
    label="Reference"
)

ax2.view_init(elev=24, azim=-58)

ax2.set_title("(b) Hybrid SNN Controlled Trajectory", pad=16)
ax2.set_xlabel("x", labelpad=8)
ax2.set_ylabel("y", labelpad=8)
ax2.set_zlabel("z", labelpad=8)

ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

ax2.legend(loc="upper left", frameon=True)

# -----------------------------------------------------
# Layout
# -----------------------------------------------------
plt.tight_layout(pad=2.5, w_pad=3.0)

# Move subplot (b) slightly left
pos = ax2.get_position()
ax2.set_position([
    pos.x0 - 0.1,   # shift left (increase magnitude if needed)
    pos.y0,
    pos.width,
    pos.height
])

# Save publication quality
plt.savefig(
    "Lorenzo_SNN.jpg",
    format="jpg",
    dpi=700,
    bbox_inches="tight"
)

plt.show()