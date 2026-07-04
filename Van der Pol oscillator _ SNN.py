import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# Van der Pol system (ground truth)
# =========================================================
def vdp_step(x, y, mu=2.0, dt=0.01):
    dx = y
    dy = mu * (1 - x**2) * y - x

    x_next = x + dt * dx
    y_next = y + dt * dy
    return x_next, y_next


# =========================================================
# Surrogate spike function (FIX for autograd)
# =========================================================
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v)
        return (v > thr).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        grad = grad_output * torch.exp(-v.abs())
        return grad, None


spike_fn = SpikeFn.apply


# =========================================================
# Bernoulli spike encoder (NEW)
# =========================================================
def bernoulli_encode(x, T, x_max=1.0):
    """
    Converts continuous input x into spike trains over time T.
    x: (batch, features)
    returns: (T, batch, features)
    """
    p = torch.clamp(x / x_max, 0, 1)

    spikes = []
    for _ in range(T):
        s = torch.bernoulli(p)
        spikes.append(s)

    return torch.stack(spikes, dim=0)


# =========================================================
# LIF Layer (stateful)
# =========================================================
class LIFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, beta=0.9, thr=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.beta = beta
        self.thr = thr

        self.v = None
        self.s = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.fc.out_features, device=device)
        self.s = torch.zeros_like(self.v)

    def forward(self, I):
        if self.v is None:
            self.init_state(I.size(0), I.device)

        # membrane update
        self.v = self.beta * (self.v - (1 - self.s)) + self.fc(I)

        # surrogate spike
        self.s = spike_fn(self.v, self.thr)

        return self.s


# =========================================================
# Multi-layer SNN
# =========================================================
class SNN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        self.layers = nn.ModuleList([
            LIFLayer(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])

    def reset(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    # =====================================================
    # MODIFIED FOR BERNoulli SPIKE INPUT
    # =====================================================
    def forward(self, x, T=10):
        batch = x.size(0)
        device = x.device

        self.reset(batch, device)

        # 🔥 NEW: spike encoding of input
        x_spikes = bernoulli_encode(x, T).to(device)

        spike_sum = 0

        for t in range(T):
            I = x_spikes[t]

            for layer in self.layers[:-1]:
                I = layer(I)

            out = self.layers[-1](I)
            spike_sum = spike_sum + out

        return spike_sum / T


# =========================================================
# Dataset (Van der Pol)
# =========================================================
def generate_trajectory(T=200, mu=1.0, dt=0.01):
    x = 0.5
    y = 0.5

    traj = []

    for _ in range(T):
        x_next, y_next = vdp_step(x, y, mu, dt)

        traj.append([x, y, x_next, y_next])

        x, y = x_next, y_next

    return torch.tensor(traj, dtype=torch.float32)


def sample_batch(batch_size=64):
    x = torch.rand(batch_size) * 2 - 1
    y = torch.rand(batch_size) * 2 - 1

    x_next, y_next = vdp_step(x, y)

    inp = torch.stack([x, y], dim=1)
    tgt = torch.stack([x_next, y_next], dim=1)

    return inp.float(), tgt.float()


def sample_batch_from_traj(traj, batch_size=64):
    idx = torch.randint(0, len(traj), (batch_size,))

    batch = traj[idx]

    x = batch[:, :2]
    y = batch[:, 2:]

    return x, y


# =========================================================
# Model
# =========================================================
layer_sizes = [2, 150, 30, 2]
model = SNN(layer_sizes)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# =========================================================
# TRAINING LOOP
# =========================================================
epochs = 200
train_loss = []

for epoch in range(epochs):
    x, y = sample_batch()

    pred = model(x, T=15)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

    if epoch % 20 == 0:
        print(f"[Train] Epoch {epoch}, Loss: {loss.item():.6f}")


# =========================================================
# TEST ROLLOUT
# =========================================================
def test_model(model, steps=200):
    model.eval()

    x = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    traj_pred = []
    traj_true = []

    with torch.no_grad():
        for _ in range(steps):
            x = model(x, T=15)
            traj_pred.append(x.squeeze().numpy())

    x_true = torch.tensor([0.5, 0.5], dtype=torch.float32)

    for _ in range(steps):
        x_true = torch.tensor(vdp_step(x_true[0], x_true[1]))
        traj_true.append(x_true.numpy())

    return traj_pred, traj_true


traj_pred, traj_true = test_model(model)


# =========================================================
# ERROR COMPUTATION
# =========================================================
traj_pred = torch.tensor(traj_pred)
traj_true = torch.tensor(traj_true)

error = torch.norm(traj_pred - traj_true, dim=1)


# =========================================================
# PLOTS
# =========================================================

plt.figure()
plt.plot(train_loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid()
plt.show()


plt.figure()
plt.plot(traj_true[:, 0], label="True x")
plt.plot(traj_pred[:, 0], label="Pred x")
plt.plot(traj_true[:, 1], '--', label="True y")
plt.plot(traj_pred[:, 1], '--', label="Pred y")
plt.title("Van der Pol Tracking with SNN (Bernoulli Input Encoding)")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(error.numpy())
plt.title("Test Tracking Error")
plt.xlabel("Time step")
plt.ylabel("||error||")
plt.grid()
plt.show()