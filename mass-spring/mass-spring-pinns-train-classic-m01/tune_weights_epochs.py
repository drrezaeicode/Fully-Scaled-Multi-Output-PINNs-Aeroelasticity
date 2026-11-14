"""
Standalone tuner: ONLY varies loss_weights_tensor and Adam epoch count.
Does NOT modify mass_spring_pinns_train_classic_r1.py
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


def ode_system(t, Y, M_mat, K_mat, k3, e1, e2):
    y = Y[:2]
    y_dot = Y[2:]
    y_ddot = np.linalg.solve(M_mat, -(K_mat @ y + (e1 - e2) * k3 * ((y[0] - y[1]) ** 3)))
    return np.concatenate([y_dot, y_ddot])


def exact_solution(t, t_exact, u_exact):
    return np.array(
        [np.interp(t, t_exact.ravel(), u_exact[:, i].ravel()) for i in range(u_exact.shape[1])]
    ).T


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


def ode_loss(model, t, M_torch, K_torch, k3_torch, e1_torch, e2_torch):
    y = model(t)
    dy_dt = torch.cat(
        [
            torch.autograd.grad(
                y[:, i],
                t,
                grad_outputs=torch.ones_like(y[:, i]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            for i in range(y.shape[1])
        ],
        dim=1,
    )
    d2y_dt2 = torch.cat(
        [
            torch.autograd.grad(
                dy_dt[:, i],
                t,
                grad_outputs=torch.ones_like(dy_dt[:, i]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            for i in range(dy_dt.shape[1])
        ],
        dim=1,
    )
    nonlinearterm = ((y[:, 0:1] - y[:, 1:2]) ** 3).reshape(-1, 1) * (
        e1_torch.reshape(1, -1) - e2_torch.reshape(1, -1)
    ) * k3_torch
    res = d2y_dt2 @ M_torch.T + y @ K_torch.T + nonlinearterm
    loss = torch.mean(res.pow(2), dim=0)
    return loss


def L2_relative_error(model, t_test_tensor, u_test_tensor):
    u_pred_tensor = model(t_test_tensor)
    err = torch.norm(u_pred_tensor - u_test_tensor) / torch.norm(u_test_tensor)
    return err


def data_loss(model, t_data_tensor, u_data_tensor):
    u_pred_tensor = model(t_data_tensor)
    loss = torch.mean((u_pred_tensor - u_data_tensor).pow(2), dim=0)
    return loss


def ic_loss(model, t, z0_true, weights=None):
    t = t.requires_grad_()
    z = model(t)
    z0 = z[0, 0]
    z1 = z[0, 1]
    dz0 = torch.autograd.grad(z0, t, grad_outputs=torch.ones_like(z0), create_graph=True, retain_graph=True)[0]
    dz1 = torch.autograd.grad(z1, t, grad_outputs=torch.ones_like(z1), create_graph=True, retain_graph=True)[0]
    z_pred = torch.cat([z0.view(1), z1.view(1), dz0.view(1), dz1.view(1)], dim=0)
    loss_ic = (z_pred - z0_true).pow(2)
    return loss_ic


def build_data():
    m, k1, k2, k3 = 1, 1, 1, 1
    z0 = [5, 1, 0, 0]
    t_span = [0, 6]
    exact_n, data_n, test_n, coll_n = 6000, 15, 2000, 1000

    M_mat = np.array([[m, 0], [0, m]])
    K_mat = np.array([[k1 + k2, -k2], [-k2, k1 + k2]])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    M_torch = torch.tensor(M_mat, dtype=torch.float32, device=device)
    K_torch = torch.tensor(K_mat, dtype=torch.float32, device=device)
    e1_torch = torch.tensor(e1, dtype=torch.float32, device=device)
    e2_torch = torch.tensor(e2, dtype=torch.float32, device=device)
    k3_torch = torch.tensor(k3, dtype=torch.float32, device=device)
    z0_tensor = torch.tensor(z0, dtype=torch.float32, device=device)

    t_exact = np.linspace(*t_span, exact_n, dtype=np.float64)
    sol = solve_ivp(
        ode_system,
        t_span,
        z0,
        args=(M_mat, K_mat, k3, e1, e2),
        t_eval=t_exact,
        method="BDF",
        rtol=1e-7,
        atol=1e-7,
    )
    t_exac = sol.t.reshape(-1, 1)
    u_exac = sol.y.T[:, :2]

    t_data = np.linspace(*t_span, data_n, dtype=np.float64).reshape(-1, 1)
    t_test = np.linspace(*t_span, test_n, dtype=np.float64).reshape(-1, 1)
    t_coll = np.linspace(*t_span, coll_n, dtype=np.float64).reshape(-1, 1)
    t_init = np.array([[t_span[0]]])

    u_data = exact_solution(t_data, t_exac, u_exac).squeeze(0)
    u_test = exact_solution(t_test, t_exac, u_exac).squeeze(0)

    t_data_tensor = torch.tensor(t_data, dtype=torch.float32, device=device)
    t_test_tensor = torch.tensor(t_test, dtype=torch.float32, device=device)
    t_coll_tensor = torch.tensor(t_coll, dtype=torch.float32, device=device, requires_grad=True)
    t_ini_tensor = torch.tensor(t_init, dtype=torch.float32, device=device)
    u_data_tensor = torch.tensor(u_data, dtype=torch.float32, device=device)
    u_test_tensor = torch.tensor(u_test, dtype=torch.float32, device=device)

    return dict(
        M_torch=M_torch,
        K_torch=K_torch,
        e1_torch=e1_torch,
        e2_torch=e2_torch,
        k3_torch=k3_torch,
        z0_tensor=z0_tensor,
        t_data_tensor=t_data_tensor,
        t_test_tensor=t_test_tensor,
        t_coll_tensor=t_coll_tensor,
        t_ini_tensor=t_ini_tensor,
        u_data_tensor=u_data_tensor,
        u_test_tensor=u_test_tensor,
    )


def train_once(weights, n_epochs, data, run_lbfgs=True, max_lbfgs_iter=500, log_every=5000):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = PINNs_MLPs([1, 128, 128, 2]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_weights_tensor = loss_weights_tensor / loss_weights_tensor.sum()

    best_l2 = float("inf")
    best_epoch = -1
    last_l2 = float("inf")

    for i in range(n_epochs):
        optimiser.zero_grad(set_to_none=True)
        data_l = data_loss(model, data["t_data_tensor"], data["u_data_tensor"])
        ode = ode_loss(
            model,
            data["t_coll_tensor"],
            data["M_torch"],
            data["K_torch"],
            data["k3_torch"],
            data["e1_torch"],
            data["e2_torch"],
        )
        ic = ic_loss(model, data["t_ini_tensor"], data["z0_tensor"])
        loss_raw = torch.stack([ode[0], ode[1], ic[0], ic[1], ic[2], ic[3], data_l[0], data_l[1]])
        loss_total = torch.sum(loss_weights_tensor * loss_raw)
        loss_total.backward()
        optimiser.step()

        if i % 200 == 0 or i == n_epochs - 1:
            with torch.no_grad():
                l2er = L2_relative_error(model, data["t_test_tensor"], data["u_test_tensor"]).item()
            last_l2 = l2er
            if l2er < best_l2:
                best_l2 = l2er
                best_epoch = i
            if i % log_every == 0:
                print(f"  Epoch {i:6d}  L2E={l2er:.6f}  Loss={loss_total.item():.4e}")

    adam_final_l2 = last_l2
    adam_best_l2 = best_l2

    lbfgs_final_l2 = None
    if run_lbfgs:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            data_l = data_loss(model, data["t_data_tensor"], data["u_data_tensor"])
            ode = ode_loss(
                model,
                data["t_coll_tensor"],
                data["M_torch"],
                data["K_torch"],
                data["k3_torch"],
                data["e1_torch"],
                data["e2_torch"],
            )
            ic = ic_loss(model, data["t_ini_tensor"], data["z0_tensor"])
            loss_raw = torch.stack([ode[0], ode[1], ic[0], ic[1], ic[2], ic[3], data_l[0], data_l[1]])
            loss_total = torch.sum(loss_weights_tensor * loss_raw)
            loss_total.backward()
            return loss_total

        for lbfgs_iter in range(max_lbfgs_iter):
            optimizer_lbfgs.step(closure)
            if (lbfgs_iter + 1) % 100 == 0:
                with torch.no_grad():
                    l2er = L2_relative_error(model, data["t_test_tensor"], data["u_test_tensor"]).item()
                print(f"  [LBFGS] {lbfgs_iter + 1:4d}  L2E={l2er:.6e}")

        with torch.no_grad():
            lbfgs_final_l2 = L2_relative_error(model, data["t_test_tensor"], data["u_test_tensor"]).item()
            if lbfgs_final_l2 < best_l2:
                best_l2 = lbfgs_final_l2

    return {
        "adam_final_l2": adam_final_l2,
        "adam_best_l2": adam_best_l2,
        "adam_best_epoch": best_epoch,
        "lbfgs_final_l2": lbfgs_final_l2,
        "best_l2": best_l2 if lbfgs_final_l2 is None else min(adam_best_l2, lbfgs_final_l2),
    }


# Candidate weights from your comments + a few sensible neighbors
WEIGHT_CANDIDATES = [
    [0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0.1, 0.1, 0.01, 0.01, 15, 15],
    [1, 1, 0.1, 0.1, 0.01, 0.01, 20, 20],
    [1, 1, 0.1, 0.1, 0.01, 0.01, 40, 40],
    [1, 1, 0.1, 0.1, 0.01, 0.01, 50, 50],
    [0.1, 0.1, 0.1, 0.1, 0.005, 0.05, 150, 150],
    [0.015, 0.015, 1500.0, 1.0, 10.0, 5.0, 1.0, 1.0],
    [0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0],
    [0.015, 0.015, 1000, 1000, 10.0, 10.0, 50.0, 50.0],
    [0.01, 0.01, 10, 10, 1, 1, 100, 100],
    [0.05, 0.05, 1, 1, 0.1, 0.1, 50, 50],
    [0.1, 0.1, 1, 1, 0.1, 0.1, 20, 20],
    [1, 1, 1, 1, 0.1, 0.1, 10, 10],
    [0.001, 0.001, 100, 100, 10, 10, 50, 50],
    [0.02, 0.02, 50, 50, 5, 5, 80, 80],
]

def save_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    out_dir = Path(__file__).resolve().parent
    results_csv = out_dir / "tune_results.csv"
    best_json = out_dir / "tune_best.json"

    data = build_data()
    rows = []

    # Phase 1: all weights @ 30k Adam (same default as your script)
    print("\n=== PHASE 1: screen all weights @ 30000 Adam ===")
    for w in WEIGHT_CANDIDATES:
        print(f"\n>>> w={w} epochs=30000")
        t0 = time.time()
        res = train_once(w, 30000, data, run_lbfgs=False, log_every=10000)
        elapsed = time.time() - t0
        row = {
            "weights": w,
            "epochs": 30000,
            "phase": "adam_screen_30k",
            "adam_final_l2": res["adam_final_l2"],
            "adam_best_l2": res["adam_best_l2"],
            "adam_best_epoch": res["adam_best_epoch"],
            "lbfgs_final_l2": None,
            "score_l2": res["adam_final_l2"],
            "elapsed_s": elapsed,
        }
        rows.append(row)
        print(
            f"    final_L2={res['adam_final_l2']:.6e}  "
            f"best_L2={res['adam_best_l2']:.6e}@{res['adam_best_epoch']}  "
            f"time={elapsed:.1f}s"
        )
        save_rows(results_csv, rows)

    # Phase 2: top-5 weights try more epoch budgets
    top_weights = [r["weights"] for r in sorted(rows, key=lambda x: x["score_l2"])[:5]]
    epoch_refine = [10000, 50000, 100000]
    print("\n=== PHASE 2: refine top-5 weights with other epoch counts ===")
    for w in top_weights:
        for epochs in epoch_refine:
            print(f"\n>>> w={w} epochs={epochs}")
            t0 = time.time()
            res = train_once(w, epochs, data, run_lbfgs=False, log_every=20000)
            elapsed = time.time() - t0
            row = {
                "weights": w,
                "epochs": epochs,
                "phase": "adam_refine",
                "adam_final_l2": res["adam_final_l2"],
                "adam_best_l2": res["adam_best_l2"],
                "adam_best_epoch": res["adam_best_epoch"],
                "lbfgs_final_l2": None,
                "score_l2": res["adam_final_l2"],
                "elapsed_s": elapsed,
            }
            rows.append(row)
            print(
                f"    final_L2={res['adam_final_l2']:.6e}  "
                f"best_L2={res['adam_best_l2']:.6e}@{res['adam_best_epoch']}  "
                f"time={elapsed:.1f}s"
            )
            save_rows(results_csv, rows)

    # Phase 3: full pipeline (Adam + LBFGS) on overall top-5 Adam finals
    ranked = sorted(
        [r for r in rows if r["phase"] in ("adam_screen_30k", "adam_refine")],
        key=lambda r: r["score_l2"],
    )[:5]
    print("\n=== PHASE 3: Full Adam + LBFGS on top-5 ===")
    final_rows = []
    for r in ranked:
        w, epochs = r["weights"], r["epochs"]
        print(f"\n>>> FULL w={w} epochs={epochs}")
        t0 = time.time()
        res = train_once(w, epochs, data, run_lbfgs=True, max_lbfgs_iter=500, log_every=20000)
        elapsed = time.time() - t0
        score = res["lbfgs_final_l2"] if res["lbfgs_final_l2"] is not None else res["adam_final_l2"]
        row = {
            "weights": w,
            "epochs": epochs,
            "phase": "full_adam_lbfgs",
            "adam_final_l2": res["adam_final_l2"],
            "adam_best_l2": res["adam_best_l2"],
            "adam_best_epoch": res["adam_best_epoch"],
            "lbfgs_final_l2": res["lbfgs_final_l2"],
            "score_l2": score,
            "elapsed_s": elapsed,
        }
        final_rows.append(row)
        rows.append(row)
        print(
            f"    adam_final={res['adam_final_l2']:.6e}  "
            f"lbfgs_final={res['lbfgs_final_l2']:.6e}  time={elapsed:.1f}s"
        )
        save_rows(results_csv, rows)

    best = min(final_rows, key=lambda r: r["score_l2"])
    print("\n========== BEST ==========")
    print(f"loss_weights_tensor = {best['weights']}")
    print(f"epochs (Adam)       = {best['epochs']}")
    print(f"L2 (after LBFGS)    = {best['score_l2']:.6e}")
    print(f"Adam final L2       = {best['adam_final_l2']:.6e}")

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"\nSaved: {results_csv}")
    print(f"Saved: {best_json}")


if __name__ == "__main__":
    main()
