import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
from snntorch import functional as SF


# =========================
# 1. Data: sinus
# =========================
T = 500
t = torch.linspace(0, 4 * torch.pi, T)

y = 1*torch.sin(t)
y = y.unsqueeze(1).unsqueeze(2)  # [T, B=1, D=1]

# =========================
# 2. Delta encoding (pure SNN input)
# =========================
spikes = spikegen.delta(y)

# =========================
# 3. Multi-layer Pure SNN
# =========================
class PureSNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Linear layers
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 5)
        self.fc3 = nn.Linear(5, 1)

        # Different surrogate gradients (your requirement 🔥)
        self.lif1 = snn.Leaky(
            beta=0.4,
            spike_grad=surrogate.fast_sigmoid()
        )

        self.lif2 = snn.Leaky(
            beta=0.6,
            spike_grad=surrogate.sigmoid()
        )

        self.lif3 = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.triangular()
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        mem_rec = []

        for t in range(x.size(0)):
            cur = x[t]

            cur = self.fc1(cur)
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.fc2(mem1)   # 🔥 membrane forward
            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.fc3(mem2)
            spk3, mem3 = self.lif3(cur, mem3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)

# =========================
# 4. Model + optimizer
# =========================
model = PureSNN()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ Use standard regression loss
loss_fn = nn.MSELoss()


# =========================
# 5. Training
# =========================
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad(set_to_none=True)

    spk_rec, mem_rec = model(spikes)

    # ✅ compare membrane with real signal
    loss = loss_fn(mem_rec, y)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# =========================
# 6. Test
# =========================
with torch.no_grad():
    _, mem_rec = model(spikes)

# =========================
# 7. Plot
# =========================
plt.plot(t.numpy(), y.squeeze().numpy(), label="True")
plt.plot(t.numpy(), mem_rec.squeeze().numpy(), "--", label="SNN")
plt.legend()
plt.title("Pure SNN (membrane output)")
plt.grid()
plt.show()