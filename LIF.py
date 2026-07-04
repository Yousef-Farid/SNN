import numpy as np
import matplotlib.pyplot as plt

# =========================
# LIF neuron
# =========================
def lif(x, u, perturb=0.0):
    beta = 0.9
    W = 0.5
    theta = 1.0

    # membrane update + perturbation
    u = beta * u + W * x + perturb

    # spike
    s = 1.0 if u >= theta else 0.0

    # reset
    u = u - s * theta

    return s, u


# =========================
# Simulation
# =========================
T = 50
x = 0.3  # constant input

# storage
U_clean, U_const, U_noise = [], [], []
u_clean = u_const = u_noise = 0.0

for t in range(T):
    # 1. No perturbation
    s, u_clean = lif(x, u_clean, perturb=0.0)
    U_clean.append(u_clean)

    # 2. Constant perturbation (+0.1)
    s, u_const = lif(x, u_const, perturb=0.1)
    U_const.append(u_const)

    # 3. Gaussian perturbation
    noise = np.random.normal(0, 0.3)
    s, u_noise = lif(x, u_noise, perturb=noise)
    U_noise.append(u_noise)


# =========================
# Plot
# =========================
plt.figure(figsize=(8,5))

plt.plot(U_clean, label="No perturbation", linewidth=2)
plt.plot(U_const, label="Constant perturbation (+0.1)", linestyle="--")
plt.plot(U_noise, label="Gaussian noise", alpha=0.8)

plt.axhline(1.0, linestyle=":", color="black", label="Threshold")

plt.xlabel("Time step")
plt.ylabel("Membrane potential")
plt.title("LIF Dynamics with and without Perturbation")
plt.legend()
plt.grid()

plt.show()