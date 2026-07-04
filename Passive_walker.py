import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


# =========================
# Simulation
# =========================

state_space = []
time = []
impacts = []

last_impact = 0
gait_period = []

current_time = T0
t_start = T0

while current_time < SIMULATION_TIME:

    sol = solve_ivp(
        equations_of_motion,
        [t_start, SIMULATION_TIME],
        x0,
        events=impact_event,
        max_step=0.01
    )

    tout = sol.t
    xout = sol.y.T

    # Wrap angles
    xout[:, 0] = wrap_to_pi(xout[:, 0])
    xout[:, 1] = wrap_to_pi(xout[:, 1])

    # Store data
    time.extend(tout)
    state_space.extend(xout)

    # Handle impact
    if sol.t_events[0].size > 0:

        impact_time = sol.t_events[0][-1]

        impact_index = len(time) - 1

        impacts.append([impact_time, impact_index])

        gait_period.append(impact_time - last_impact)

        last_impact = impact_time

        x0 = impact_map(xout[-1])

        t_start = impact_time

    else:
        break

    current_time = time[-1]

# Convert to arrays
time = np.array(time)
state_space = np.array(state_space)
impacts = np.array(impacts)

# =========================
# Plot Results
# =========================

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    state_space[:, 0],
    state_space[:, 2],
    state_space[:, 1]
)

ax.set_xlabel("q1")
ax.set_ylabel("Dq1")
ax.set_zlabel("q2")

plt.title("State Space Trajectory")
plt.show()