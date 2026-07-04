import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import nengo
from matplotlib import rcParams


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"

rcParams["font.size"] = 18
rcParams["axes.labelsize"] = 18
rcParams["axes.titlesize"] = 20
rcParams["legend.fontsize"] = 15
rcParams["lines.linewidth"] = 2.2
rcParams["savefig.dpi"] = 700

# =========================
# Global Parameters
# =========================

mH = 10
m1 = 5
m2 = 5
a = 0.5
b = 0.5
l = a + b
g = 9.81

p1 = mH * l**2 + m1 * b**2 + m2 * l**2
p2 = m2 * l * a
p3 = m2 * a**2
p4 = (m1 * b + m2 * l + mH * l) * g
p5 = m2 * a * g

p6 = m1 * l**2 + mH * l**2 + m2 * b**2
p7 = m1 * a * l
p8 = m1 * a**2
p9 = m1 * a * b
p10 = m2 * b * l + mH * l**2 + m1 * b * l
p11 = m2 * b * a

psi = 4.1255 * np.pi / 180
# psi = 3.1 * np.pi / 180

# =========================
# Initial Conditions
# =========================

q0 = np.array([
    12.53 * np.pi / 180,
    -18.53 * np.pi / 180
])

Dq0 = np.array([-1, 1.5])

x0 = np.concatenate((q0, Dq0))

SIMULATION_TIME = 15
DRAW_INTERVAL = 0.05
T0 = 0

# =====================================================
# SNN Parameters
# =====================================================

dt = 0.001
tau = 0.05

# =========================
# Helper Functions
# =========================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =========================
# Equations of Motion
# =========================

def equations_of_motion(t, x):

    q1, q2, Dq1, Dq2 = x
    Dq = np.array([Dq1, Dq2])

    m11 = p1
    m12 = -p2 * np.cos(q1 - q2)
    m21 = m12
    m22 = p3

    c11 = 0
    c12 = -Dq2 * np.sin(q1 - q2) * p2
    c21 = -Dq1 * np.sin(q1 - q2) * p2
    c22 = 0

    g1 = -np.sin(q1) * p4
    g2 = np.sin(q2) * p5

    M = np.array([
        [m11, m12],
        [m21, m22]
    ])

    C = np.array([
        [c11, c12],
        [c21, c22]
    ])

    G = np.array([g1, g2])

    DDq = np.linalg.solve(M, -C @ Dq - G)

    Dx = np.concatenate((Dq, DDq))

    return Dx


# =========================
# Impact Event
# =========================

def impact_event(t, x):

    q1 = x[0]
    q2 = x[1]
    Dq2 = x[3]

    value = np.cos(q1 + psi) - np.cos(q2 + psi)

    # Stop integration only if Dq2 <= 0
    if Dq2 > 0:
        return 1.0

    return value


impact_event.terminal = True
impact_event.direction = -1


# =========================
# Impact Map
# =========================

def impact_map(x_minus):

    q1, q2, Dq1, Dq2 = x_minus

    Qp = np.array([
        [-np.cos(q1 - q2) * p7 + p6,
         p8 - np.cos(q1 - q2) * p7],

        [-np.cos(q1 - q2) * p7,
         p8]
    ])

    Qn = np.array([
        [np.cos(q1 - q2) * p10 - p9,
         -p11],

        [-p9,
         0]
    ])

    Dq_plus = np.linalg.solve(Qp, Qn @ np.array([Dq1, Dq2]))

    q_plus = np.array([q2, q1])

    x_plus = np.concatenate((q_plus, Dq_plus))

    return x_plus


# =========================
# Travel Offset
# =========================

def travel_offset(impacts, static_origins):

    offset_origins = np.hstack((
        static_origins,
        np.zeros((static_origins.shape[0], 2))
    ))

    for impact in impacts:

        impact_index = int(impact[1])

        offset_origins[impact_index + 1:, 0::2] += \
            static_origins[impact_index, 6]

        offset_origins[impact_index + 1:, 1::2] += \
            static_origins[impact_index, 7]

    return offset_origins


# =========================
# Walker Origins
# =========================

def walker_origins(q1, q2):

    o1x = -np.sin(q1) * b
    o1y = np.cos(q1) * b

    ohx = -np.sin(q1) * a - np.sin(q1) * b
    ohy = np.cos(q1) * a + np.cos(q1) * b

    o2x = -l * np.sin(q1) + a * np.sin(q2)
    o2y = l * np.cos(q1) - a * np.cos(q2)

    oex = np.sin(q2) * b - l * np.sin(q1) + a * np.sin(q2)
    oey = -np.cos(q2) * b + l * np.cos(q1) - a * np.cos(q2)

    o1 = [o1x, o1y]
    oh = [ohx, ohy]
    o2 = [o2x, o2y]
    oe = [oex, oey]

    origins = np.array(o1 + oh + o2 + oe)

    return origins

# =====================================================
# RUN REAL PASSIVE WALKER FIRST
# =====================================================

state_space = []
time = []
impacts = []

last_impact = 0
gait_period = []

current_time = T0
t_start = T0

x_init = np.concatenate((q0, Dq0))

while current_time < SIMULATION_TIME:

    sol = solve_ivp(
        equations_of_motion,
        [t_start, SIMULATION_TIME],
        x_init,
        events=impact_event,
        max_step=dt
    )

    tout = sol.t
    xout = sol.y.T

    # Wrap angles
    xout[:, 0] = wrap_to_pi(xout[:, 0])
    xout[:, 1] = wrap_to_pi(xout[:, 1])

    # Store
    time.extend(tout)
    state_space.extend(xout)

    # Handle impacts
    if sol.t_events[0].size > 0:

        impact_time = sol.t_events[0][-1]

        impact_index = len(time) - 1

        impacts.append([impact_time, impact_index])

        gait_period.append(impact_time - last_impact)

        last_impact = impact_time

        x_init = impact_map(xout[-1])

        t_start = impact_time

    else:
        break

    current_time = time[-1]

# Convert to arrays
time = np.array(time)
state_space = np.array(state_space)
impacts = np.array(impacts)

# =====================================================
# REMOVE DUPLICATE TIMES
# =====================================================

unique_indices = np.unique(time, return_index=True)[1]

time_unique = time[unique_indices]
state_unique = state_space[unique_indices]

# =====================================================
# SNN DYNAMICS
# =====================================================

def walker_dynamics(x):

    q1, q2, Dq1, Dq2 = x

    Dq = np.array([Dq1, Dq2])

    m11 = p1
    m12 = -p2 * np.cos(q1 - q2)
    m21 = m12
    m22 = p3

    M = np.array([
        [m11, m12],
        [m21, m22]
    ])

    c11 = 0
    c12 = -Dq2 * np.sin(q1 - q2) * p2
    c21 = -Dq1 * np.sin(q1 - q2) * p2
    c22 = 0

    C = np.array([
        [c11, c12],
        [c21, c22]
    ])

    g1 = -np.sin(q1) * p4
    g2 = np.sin(q2) * p5

    G = np.array([g1, g2])

    DDq = np.linalg.solve(M, -C @ Dq - G)

    dx = np.array([
        Dq1,
        Dq2,
        DDq[0],
        DDq[1]
    ])

    return dx * tau

# =====================================================
# BUILD SNN
# =====================================================

model = nengo.Network()

with model:

    state = nengo.Ensemble(
        n_neurons=10000,
        dimensions=4,
        radius=10,
        neuron_type=nengo.LIF()
    )

    init_node = nengo.Node(
        lambda t: x_init if t < 0.05 else [0, 0, 0, 0]
    )

    nengo.Connection(init_node, state, synapse=None)

    nengo.Connection(
        state,
        state,
        transform=np.eye(4),
        synapse=tau
    )

    nengo.Connection(
        state,
        state,
        function=walker_dynamics,
        synapse=tau
    )

    probe = nengo.Probe(state, synapse=0.01)

# =====================================================
# RUN SNN
# =====================================================

with nengo.Simulator(model, dt=dt) as sim:

    sim.run(SIMULATION_TIME)

snn = sim.data[probe]

snn_time = sim.trange()

# =====================================================
# INTERPOLATE REAL SYSTEM TO SAME TIME GRID
# =====================================================

real_interp = np.zeros((len(snn_time), 4))

for i in range(4):

    real_interp[:, i] = np.interp(
        snn_time,
        time_unique,
        state_unique[:, i]
    )

# =====================================================
# REMOVE INITIAL TRANSIENT
# =====================================================

warmup = int(2.0 / dt)

real_plot = real_interp[warmup:]
snn_plot = snn[warmup:]

# =====================================================
# PLOT
# =====================================================

fig = plt.figure(figsize=(11, 7))

ax = fig.add_subplot(111, projection='3d')

# Real Passive Walker
ax.plot(
    real_plot[:, 0],
    real_plot[:, 2],
    real_plot[:, 1],
    color="red",
    linewidth=2.5,
    label="Real Passive Walker"
)

# SNN Approximation
ax.plot(
    snn_plot[:, 0],
    snn_plot[:, 2],
    snn_plot[:, 1],
    "--",
    color="tab:blue",
    linewidth=2.0,
    alpha=0.95,
    label="SNN Approximation"
)

ax.set_xlabel("q1")
ax.set_ylabel("Dq1")
ax.set_zlabel("q2")

ax.grid(True, alpha=0.3)

ax.legend()

plt.title("Passive Walker vs SNN Approximation")

plt.tight_layout()

plt.show()