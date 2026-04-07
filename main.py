import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 物理パラメータ
# ===============================
E = 2   # ヤング率 [Pa]
A = 1    # 断面積 [m^2]
P = 10    # 集中荷重 [N]
L = 3.0     # 棒の長さ [m]
EA = E * A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# PINN モデル定義
# ===============================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        # Dirichlet条件 u(0)=0 を保証
        return x * self.net(x)

# ===============================
# モデル・最適化
# ===============================
model = PINN().to(device)

# ===============================
# 損失関数
# ===============================
def loss_func(x_pde, x_bc_p):
    # PDE残差 (EA * u'' = 0)
    x_pde.requires_grad = True
    u = model(x_pde)
    du_dx = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_pde, torch.ones_like(du_dx), create_graph=True)[0]
    pde_loss = torch.mean((EA * d2u_dx2)**2)

    # Neumann BC (x=L で EA*u'(L)=P)
    x_bc_p.requires_grad = True
    u_bc_p = model(x_bc_p)
    du_dx_bc_p = torch.autograd.grad(u_bc_p, x_bc_p, torch.ones_like(u_bc_p), create_graph=True)[0]
    bc_p_loss = torch.mean((EA * du_dx_bc_p - P)**2)

    return pde_loss, bc_p_loss

# ===============================
# 学習データ
# ===============================
n_pde = 50
x_pde = torch.linspace(0, L, n_pde).view(-1, 1).to(device)
x_bc_p = torch.tensor([[L]], dtype=torch.float32).to(device)

# ===============================
# 学習 (Adam → LBFGS)
# ===============================
# Step 1: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    pde_loss, bc_p_loss = loss_func(x_pde, x_bc_p)
    loss = pde_loss + 10 * bc_p_loss
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"[Adam] Epoch {epoch+1}/{epochs}, Loss={loss.item():.4e}, PDE={pde_loss.item():.4e}, BCp={bc_p_loss.item():.4e}")

# Step 2: LBFGS
#optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50000, tolerance_grad=1e-9, tolerance_change=1e-9)

def closure():
    optimizer.zero_grad()
    pde_loss, bc_p_loss = loss_func(x_pde, x_bc_p)
    loss = pde_loss + 10 * bc_p_loss
    loss.backward()
    return loss

#print("Switching to LBFGS...")
#optimizer.step(closure)

# ===============================
# 結果の可視化
# ===============================
x_plot = torch.linspace(0, L, 100).view(-1, 1).to(device)
u_pred = model(x_plot).detach().cpu().numpy()
x_plot = x_plot.detach().cpu().numpy()

# 解析解: u(x) = P*x/EA
u_analytic = P * x_plot / EA

plt.figure(figsize=(10, 6))
plt.plot(x_plot, u_pred, label='PINN Solution', color='red', linewidth=2)
plt.plot(x_plot, u_analytic, '--', label='Analytical Solution', color='blue')
plt.title('Displacement of a 1D Bar under Point Load')
plt.xlabel('Position x [m]')
plt.ylabel('Displacement u [m]')
plt.legend()
plt.grid(True)
plt.show()
