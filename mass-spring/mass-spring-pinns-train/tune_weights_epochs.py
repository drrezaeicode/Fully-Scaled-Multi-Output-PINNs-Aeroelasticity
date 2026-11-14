"""Hyperparameter search for loss weights + epochs. Does not modify the original script."""
import sys
import csv
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}", flush=True)


def ode_system(t, Y, M_mat, K_mat, k3, e1, e2):
    y = Y[:2]
    y_dot = Y[2:]
    y_ddot = np.linalg.solve(M_mat, -(K_mat @ y + (e1 - e2) * k3 * ((y[0] - y[1]) ** 3)))
    return np.concatenate([y_dot, y_ddot])


def exact_solution(t, t_exact, u_exact):
    return np.array(
        [np.interp(t, t_exact.ravel(), u_exact[:, i].ravel()) for i in range(u_exact.shape[1])]
    ).T


def ode_system_scaled(t, Y, M_bar, K_bar, K_0, k3, e1, e2, sfh, smh, sfa, sma):
    y = Y[:2]
    y_dot = Y[2:]
    nonlinear_term = k3 * ((y[0] - smh) / sfh - (y[1] - sma) / sfa) ** 3
    y_ddot = np.linalg.solve(M_bar, -(K_bar @ y + (e1 - e2) * nonlinear_term - K_0))
    return np.concatenate([y_dot, y_ddot])


class PINNs_MLPs(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh, device=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.activation = activation

    def forward(self, x, activation=None):
        if activation is None:
            activation = self.activation
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x


def ode_loss(model, t, M_bar_torch, K_bar_torch, K_0_torch, k3_torch, e1_torch, e2_torch,
             sfh_tensor, smh_tensor, sfa_tensor, sma_tensor):
    y = model(t)
    dy_dt = torch.cat([
        torch.autograd.grad(y[:, i], t, grad_outputs=torch.ones_like(y[:, i]),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        for i in range(y.shape[1])
    ], dim=1)
    d2y_dt2 = torch.cat([
        torch.autograd.grad(dy_dt[:, i], t, grad_outputs=torch.ones_like(dy_dt[:, i]),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        for i in range(dy_dt.shape[1])
    ], dim=1)
    nonlinearterm = (
        ((y[:, 0:1] - smh_tensor) / sfh_tensor - (y[:, 1:2] - sma_tensor) / sfa_tensor) ** 3
    ).reshape(-1, 1) * (e1_torch.reshape(1, -1) - e2_torch.reshape(1, -1)) * k3_torch
    res = d2y_dt2 @ M_bar_torch.T + y @ K_bar_torch.T - K_0_torch + nonlinearterm
    return torch.mean(res.pow(2), dim=0)


def L2_relative_error(model, t_test_scaled_tensor, u_test_scaled_tensor):
    u_pred = model(t_test_scaled_tensor)
    return torch.norm(u_pred - u_test_scaled_tensor) / torch.norm(u_test_scaled_tensor)


def data_loss(model, t_data_scaled_tensor, u_data_scaled_tensor):
    u_pred = model(t_data_scaled_tensor)
    return torch.mean((u_pred - u_data_scaled_tensor).pow(2), dim=0)


def ic_loss(model, t, z0_scaled):
    t = t.requires_grad_()
    z = model(t)
    z0, z1 = z[0, 0], z[0, 1]
    dz0 = torch.autograd.grad(z0, t, grad_outputs=torch.ones_like(z0), create_graph=True, retain_graph=True)[0]
    dz1 = torch.autograd.grad(z1, t, grad_outputs=torch.ones_like(z1), create_graph=True, retain_graph=True)[0]
    z_pred = torch.cat([z0.view(1), z1.view(1), dz0.view(1), dz1.view(1)], dim=0)
    return (z_pred - z0_scaled).pow(2)


def prepare_data():
    m, k1, k2, k3 = 1, 1, 1, 1
    z0 = [5, 1, 0, 0]
    t_span = [0, 6]
    exact_n, data_n, test_n, coll_n = 6000, 15, 2000, 1000

    M_mat = np.array([[m, 0], [0, m]], dtype=np.float64)
    K_mat = np.array([[k1 + k2, -k2], [-k2, k1 + k2]], dtype=np.float64)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    t_exact = np.linspace(*t_span, exact_n, dtype=np.float64)
    sol = solve_ivp(ode_system, t_span, z0, args=(M_mat, K_mat, k3, e1, e2),
                    t_eval=t_exact, method="BDF", rtol=1e-7, atol=1e-7)
    t_exac = sol.t.reshape(-1, 1)
    u_exac = sol.y.T[:, :2]

    t_data = np.linspace(*t_span, data_n, dtype=np.float64).reshape(-1, 1)
    t_test = np.linspace(*t_span, test_n, dtype=np.float64).reshape(-1, 1)
    t_coll = np.linspace(*t_span, coll_n, dtype=np.float64).reshape(-1, 1)
    t_init = np.array([[t_span[0]]])

    u_data = exact_solution(t_data, t_exac, u_exac).squeeze(0)
    u_test = exact_solution(t_test, t_exac, u_exac).squeeze(0)
    u_ini = np.array(z0[0:2], dtype=np.float32).reshape(1, -1)
    du_ini = np.array(z0[2:4], dtype=np.float32).reshape(1, -1)

    t_exac_tensor = torch.tensor(t_exac, dtype=torch.float32, device=device)
    t_data_tensor = torch.tensor(t_data, dtype=torch.float32, device=device)
    t_test_tensor = torch.tensor(t_test, dtype=torch.float32, device=device)
    t_coll_tensor = torch.tensor(t_coll, dtype=torch.float32, device=device)
    t_ini_tensor = torch.tensor(t_init, dtype=torch.float32, device=device)
    u_exac_tensor = torch.tensor(u_exac, dtype=torch.float32, device=device)
    u_data_tensor = torch.tensor(u_data, dtype=torch.float32, device=device)
    u_test_tensor = torch.tensor(u_test, dtype=torch.float32, device=device)
    u_ini_tensor = torch.tensor(u_ini, dtype=torch.float32, device=device)
    du_ini_tensor = torch.tensor(du_ini, dtype=torch.float32, device=device)

    scaler = mmsptx(t_range=(-1, 1), x_range=(-1, 1), device=device).fit(t_exac_tensor, u_exac_tensor)
    sfh = scaler.xscale_[0, 0].item()
    smh = scaler.xmin_[0, 0].item()
    sfa = scaler.xscale_[0, 1].item()
    sma = scaler.xmin_[0, 1].item()
    sft = scaler.tscale_.item()

    z0_scaled = [z0[0] * sfh + smh, z0[1] * sfa + sma, sft / sfh * z0[2], sft / sfa * z0[3]]
    z0_scaled_tensor = torch.tensor(z0_scaled, dtype=torch.float32, device=device)

    DK = np.diag((1 / scaler.xscale_).cpu().numpy().flatten())
    D0 = (scaler.xmin_ / scaler.xscale_).detach().cpu().numpy()
    M_bar = sft**2 * M_mat @ np.linalg.inv(np.diag(scaler.xscale_.cpu().numpy().flatten()))
    K_bar = K_mat @ DK
    K_0 = K_mat @ D0.ravel()

    t_exac_scaled = scaler.transformT(t_exac_tensor).detach().cpu().numpy().ravel()
    solve_ivp(ode_system_scaled, [float(t_exac_scaled[0]), float(t_exac_scaled[-1])], z0_scaled,
              args=(M_bar, K_bar, K_0, k3, e1, e2, sfh, smh, sfa, sma),
              t_eval=t_exac_scaled, method="BDF", rtol=1e-7, atol=1e-7)

    t_data_scaled_tensor, u_data_scaled_tensor = scaler.transform(t_data_tensor, u_data_tensor)
    t_test_scaled_tensor, u_test_scaled_tensor = scaler.transform(t_test_tensor, u_test_tensor)
    t_ini_scaled_tensor, _ = scaler.transform(t_ini_tensor, u_ini_tensor)
    t_ini_scaled_tensor = t_ini_scaled_tensor.requires_grad_(True)
    t_coll_scaled_tensor = scaler.transformT(t_coll_tensor).requires_grad_(True)

    return dict(
        t_data_scaled_tensor=t_data_scaled_tensor,
        u_data_scaled_tensor=u_data_scaled_tensor,
        t_test_scaled_tensor=t_test_scaled_tensor,
        u_test_scaled_tensor=u_test_scaled_tensor,
        t_ini_scaled_tensor=t_ini_scaled_tensor,
        z0_scaled_tensor=z0_scaled_tensor,
        t_coll_scaled_tensor=t_coll_scaled_tensor,
        M_bar_torch=torch.tensor(M_bar, dtype=torch.float32, device=device),
        K_bar_torch=torch.tensor(K_bar, dtype=torch.float32, device=device),
        K_0_torch=torch.tensor(K_0, dtype=torch.float32, device=device),
        k3_torch=torch.tensor(k3, dtype=torch.float32, device=device),
        e1_torch=torch.tensor(e1, dtype=torch.float32, device=device),
        e2_torch=torch.tensor(e2, dtype=torch.float32, device=device),
        sfh_tensor=torch.tensor(sfh, dtype=torch.float32, device=device),
        smh_tensor=torch.tensor(smh, dtype=torch.float32, device=device),
        sfa_tensor=torch.tensor(sfa, dtype=torch.float32, device=device),
        sma_tensor=torch.tensor(sma, dtype=torch.float32, device=device),
    )


def train_once(data, weights, epochs, l2_every=100, print_every=2000):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = PINNs_MLPs([1, 128, 128, 2]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / w.sum()

    best_l2, best_epoch, final_l2 = float("inf"), -1, None
    snapshots = {}  # epoch -> l2 for early-stop recommendation

    for i in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        data_l = data_loss(model, data["t_data_scaled_tensor"], data["u_data_scaled_tensor"])
        ode = ode_loss(
            model, data["t_coll_scaled_tensor"], data["M_bar_torch"], data["K_bar_torch"],
            data["K_0_torch"], data["k3_torch"], data["e1_torch"], data["e2_torch"],
            data["sfh_tensor"], data["smh_tensor"], data["sfa_tensor"], data["sma_tensor"],
        )
        ic = ic_loss(model, data["t_ini_scaled_tensor"], data["z0_scaled_tensor"])
        loss_raw = torch.stack([ode[0], ode[1], ic[0], ic[1], ic[2], ic[3], data_l[0], data_l[1]])
        loss_total = torch.sum(w * loss_raw)
        loss_total.backward()
        optimiser.step()

        if i % l2_every == 0 or i == epochs - 1:
            with torch.no_grad():
                l2er = L2_relative_error(
                    model, data["t_test_scaled_tensor"], data["u_test_scaled_tensor"]
                ).item()
            snapshots[i] = l2er
            if l2er < best_l2:
                best_l2, best_epoch = l2er, i
            final_l2 = l2er
            if i % print_every == 0 or i == epochs - 1:
                print(
                    f"  ep {i:5d}  loss={loss_total.item():.4e}  L2={l2er:.6f}  best={best_l2:.6f}@{best_epoch}",
                    flush=True,
                )
    return best_l2, best_epoch, final_l2, snapshots


if __name__ == "__main__":
    data = prepare_data()
    # order: ode_h, ode_a, ic_h, ic_a, ic_dh, ic_da, data_h, data_a
    configs = [
        ("baseline_10k", [0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0], 10000),
        ("baseline_20k", [0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0], 20000),
        ("baseline_30k", [0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0], 30000),
        ("ode2x_20k", [0.03, 0.03, 10, 10, 10, 10, 50, 50], 20000),
        ("ode5x_20k", [0.075, 0.075, 10, 10, 10, 10, 50, 50], 20000),
        ("ode10x_20k", [0.15, 0.15, 10, 10, 10, 10, 50, 50], 20000),
        ("data2x_20k", [0.015, 0.015, 10, 10, 10, 10, 100, 100], 20000),
        ("data05x_20k", [0.015, 0.015, 10, 10, 10, 10, 25, 25], 20000),
        ("ic2x_20k", [0.015, 0.015, 20, 20, 20, 20, 50, 50], 20000),
        ("ic05x_20k", [0.015, 0.015, 5, 5, 5, 5, 50, 50], 20000),
        ("phys_data_20k", [0.05, 0.05, 10, 10, 10, 10, 80, 80], 20000),
        ("bal_20k", [0.02, 0.02, 8, 8, 8, 8, 40, 40], 20000),
        ("strong_phys_20k", [0.2, 0.2, 15, 15, 15, 15, 40, 40], 20000),
        ("tiny_ode_20k", [0.005, 0.005, 10, 10, 10, 10, 50, 50], 20000),
        ("mid_all_25k", [0.04, 0.04, 12, 12, 12, 12, 60, 60], 25000),
    ]

    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    out_csv = "tune_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "best_l2", "best_epoch", "final_l2", "epochs", "weights"])
        f.flush()

        results = []
        for name, w, ep in configs:
            if only and name not in only:
                continue
            print(f"\n===== {name}  w={w}  epochs={ep} =====", flush=True)
            best_l2, best_ep, final_l2, _ = train_once(data, w, ep)
            row = [name, f"{best_l2:.8f}", best_ep, f"{final_l2:.8f}", ep, str(w)]
            writer.writerow(row)
            f.flush()
            results.append((best_l2, final_l2, best_ep, ep, name, w))
            print(f"RESULT {name}: best_L2={best_l2:.6f} @ep{best_ep}  final_L2={final_l2:.6f}", flush=True)

    results.sort(key=lambda x: x[0])
    print("\n========== RANKED BY BEST L2 ==========", flush=True)
    for best_l2, final_l2, best_ep, ep, name, w in results:
        print(f"{best_l2:.6f}  final={final_l2:.6f}  best_ep={best_ep}/{ep}  {name}  w={w}", flush=True)

    results.sort(key=lambda x: x[1])
    print("\n========== RANKED BY FINAL L2 ==========", flush=True)
    for best_l2, final_l2, best_ep, ep, name, w in results:
        print(f"{final_l2:.6f}  best={best_l2:.6f}@{best_ep}  {name}  w={w}  ep={ep}", flush=True)
