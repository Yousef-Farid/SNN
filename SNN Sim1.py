import nengo
import numpy as np
import matplotlib.pyplot as plt

model = nengo.Network()

with model:
    stim = nengo.Node(lambda t: np.sin(2*t))

    pre = nengo.Ensemble(2500, 1)
    post = nengo.Ensemble(2500, 1)

    nengo.Connection(stim, pre)

    conn = nengo.Connection(
        pre,
        post,
        function=lambda x: x,     # better starting guess
        learning_rule_type=nengo.PES(learning_rate=1e-6)
    )

    error = nengo.Node(size_in=1)

    nengo.Connection(stim, error, transform=1)
    nengo.Connection(post, error, transform=-1, synapse=0.05)

    nengo.Connection(error, conn.learning_rule, synapse=0.1)

    inp = nengo.Probe(stim)
    out = nengo.Probe(post, synapse=0.005)
    err = nengo.Probe(error, synapse=0.05)

with nengo.Simulator(model) as sim:
    sim.run(5)

t = sim.trange()

plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(t, sim.data[inp], label="Input")
plt.plot(t, sim.data[out], label="Output")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, sim.data[err])
plt.title("Error")
plt.grid()

plt.tight_layout()
plt.show()