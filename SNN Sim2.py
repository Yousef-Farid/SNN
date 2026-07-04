import nengo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =====================================================
# IEEE CSL / JOURNAL QUALITY FIGURE SETTINGS
# =====================================================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"

rcParams["font.size"] = 18
rcParams["axes.titlesize"] = 20
rcParams["axes.labelsize"] = 20
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["legend.fontsize"] = 18
rcParams["axes.linewidth"] = 1.2
rcParams["lines.linewidth"] = 2.2
rcParams["savefig.dpi"] = 700

# ---------------------------------
# Parameters
# ---------------------------------
mu = 2.0
dt = 0.001
tau = 0.05
sim_time = 20.0

# ---------------------------------
# Real Van der Pol System
# ---------------------------------
def vdp_real_step(state):
    x, y = state
    dx = y
    dy = mu * (1 - x**2) * y - x
    return np.array([x + dt * dx, y + dt * dy])

# ---------------------------------
# Nengo Dynamics
# ---------------------------------
def vdp_dyn(state):
    x, y = state
    dx = y
    dy = mu * (1 - x**2) * y - x
    return np.array([dx, dy]) * tau

# ---------------------------------
# Build SNN Model
# ---------------------------------
model = nengo.Network()

with model:

    state = nengo.Ensemble(
        n_neurons=5000,
        dimensions=2,
        radius=5,
        neuron_type=nengo.LIF()
    )

    init = nengo.Node(lambda t: [2.0, 0.5] if t < 0.05 else [0, 0])
    nengo.Connection(init, state, synapse=None)

    nengo.Connection(
        state,
        state,
        transform=np.eye(2),
        synapse=tau
    )

    nengo.Connection(
        state,
        state,
        function=vdp_dyn,
        synapse=tau
    )

    probe = nengo.Probe(state, synapse=0.02)

# ---------------------------------
# Run SNN
# ---------------------------------
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(sim_time)

snn = sim.data[probe]

# ---------------------------------
# Run Real System
# ---------------------------------
steps = int(sim_time / dt)

real = np.zeros((steps, 2))
real[0] = [2.0, 0.5]

for i in range(steps - 1):
    real[i + 1] = vdp_real_step(real[i])

# ---------------------------------
# Remove Warmup
# ---------------------------------
warmup = int(3.0 / dt)
snn_clean = snn[warmup:]

N = min(len(real), len(snn_clean))
real = real[:N]
snn_clean = snn_clean[:N]

# ---------------------------------
# PUBLICATION QUALITY FIGURE
# ---------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5))

# Real trajectory
ax.plot(
    real[:,0],
    real[:,1],
    color="red",
    linewidth=2.4,
    label="Van der Pol System"
)

# SNN trajectory
ax.plot(
    snn_clean[:,0],
    snn_clean[:,1],
    "--",
    color="tab:blue",
    linewidth=2.0,
    alpha=0.95,
    label="SNN Approximation"
)

# Labels
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

# Grid
ax.grid(True, alpha=0.28)

# Legend
ax.legend(loc="best", frameon=True)

# Better frame
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

# ---------------------------------
# Save High Quality
# ---------------------------------
plt.savefig(
    "VanderPol_SNN.jpg",
    format="jpg",
    dpi=700,
    bbox_inches="tight"
)

plt.show()
