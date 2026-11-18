# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 02:20:37 2026

@author: Mohsen
"""

# ======================================================================
# 1) Import the training module FIRST (it may wipe the namespace)
# ======================================================================
import sys, os
sys.path.append(os.path.abspath("../pitch-plunge-2d-u10-pinns-train"))
from Pitch_Plunge_2D_U_10_pinns_train_linspace import PINNs_MLPs
_PINN = PINNs_MLPs

# ======================================================================
# 2) Re-import everything
# ======================================================================
import sys, os, gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx   # My Library

PINNs_MLPs = _PINN
plt.close("all")
gc.collect()
device = "cpu"
os.makedirs("figures", exist_ok=True)

DIR = "../pitch-plunge-2d-u10-pinns-train"


# ======================================================================
# Helper: load workspace into a dict (nothing touches globals)
# ======================================================================
def load_ws(path):
    ws = torch.load(path, map_location="cpu", weights_only=False)
    plt.close("all")
    bad = [k for k, v in ws.items()
           if "matplotlib" in type(v).__module__
           or (isinstance(v, np.ndarray) and v.dtype == object)
           or (isinstance(v, (list, tuple)) and len(v) > 0
               and "matplotlib" in type(v[0]).__module__)]
    for k in bad:
        ws.pop(k)
    gc.collect()
    print(f"loaded {len(ws)} items from {path}  (dropped: {bad})")
    return ws


# ======================================================================
# Load the four sampling strategies
# ======================================================================
methods_files = {          # key : (workspace file, network file)
    "linspace": ("pitch_plunge_2D_U_10_pinns_train_linspace.pth",
                 "pitch_plunge_2D_U_10_pinns_network_linspace.pth"),
    "random":   ("pitch_plunge_2D_U_10_pinns_train_random.pth",
                 "pitch_plunge_2D_U_10_pinns_network_random.pth"),
    "lhs":      ("pitch_plunge_2D_U_10_pinns_train_lhs.pth",
                 "pitch_plunge_2D_U_10_pinns_network_lhs.pth"),
    "sobol":    ("pitch_plunge_2D_U_10_pinns_train_sobol.pth",
                 "pitch_plunge_2D_U_10_pinns_network_sobol.pth"),
}

U_pred, dU_pred, T_data = {}, {}, {}
scaler = None

for key, (f_ws, f_net) in methods_files.items():
    ws = load_ws(f"{DIR}/{f_ws}")

    # common (identical across runs) tensors: taken from the last workspace, as before
    t_exac_tensor     = ws["t_exac_tensor"]
    u_exac_tensor     = ws["u_exac_tensor"]
    u_exac_dot_tensor = ws["u_exac_dot_tensor"]
    t_test_tensor     = ws["t_test_tensor"]
    T_data[key]       = ws["t_data_tensor"]

    if scaler is None:
        scaler = mmsptx(t_range=(-1, 1), x_range=(-1, 1), device=device).fit(
            t_exac_tensor, u_exac_tensor)

    model = PINNs_MLPs([1, 128, 128, 2]).to(device)
    model.load_state_dict(torch.load(f"{DIR}/{f_net}", map_location=device, weights_only=True))
    model.eval()

    t_s = ws["t_test_scaled_tensor"].detach().clone().requires_grad_(True)
    u_s = model(t_s)                                   # (N,2)
    du_s = torch.zeros_like(u_s)
    for k in range(2):
        du_s[:, k:k+1] = torch.autograd.grad(
            u_s[:, k:k+1], t_s,
            grad_outputs=torch.ones_like(u_s[:, k:k+1]),
            create_graph=True)[0]

    U_pred[key]  = scaler.inverse_transformX(u_s).detach()
    dU_pred[key] = scaler.scale_first_derivative_inverse_transform(du_s).detach()

n  = lambda x: x.detach().cpu().numpy()
t_te = n(t_test_tensor).ravel()

# ======================================================================
# Journal-accurate setup
# ======================================================================
plt.close("all")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.grid": True,
    "grid.color": "0.8",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")


def tag(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold")


# ======================================================================
# Errors
# ======================================================================
err  = {k: n(u_exac_tensor     - U_pred[k])  for k in U_pred}
derr = {k: n(u_exac_dot_tensor - dU_pred[k]) for k in U_pred}

STY = [("linspace", "UG",    "#009E73", "-",  1),
       ("random",   "UR",    "#0072B2", "--", 2),
       ("lhs",      "LHS",   "#D55E00", "-.", 3),
       ("sobol",    "Sobol", "#4D4D4D", ":",  4)]


def error_panel(ax, E, col, ylab, ylim=None):
    for key, lab, c, ls, z in STY:
        ax.plot(t_te, E[key][:, col], label=lab, color=c, lw=1.4, ls=ls, zorder=z)
    ax.set_xlim([0, 2])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$t\,(\mathrm{s})$")
    ax.set_ylabel(ylab)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


# ======================================================================
# Figure 1 : error histories, 2 x 2 subplots, one shared legend
# ======================================================================
fig_1, axs1 = plt.subplots(2, 2, figsize=(7.0, 6.0))

error_panel(axs1[0, 0], err,  0, r"$e_h$")
error_panel(axs1[0, 1], err,  1, r"$e_\alpha$",         (-.001, .001))
error_panel(axs1[1, 0], derr, 0, r"$e_{\dot{h}}$",      (-.002, .002))
error_panel(axs1[1, 1], derr, 1, r"$e_{\dot{\alpha}}$", (-.06, .06))

for ax, s in zip(axs1.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
    tag(ax, s)

fig_1.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.5, w_pad=1.5)
h, l = axs1[0, 0].get_legend_handles_labels()
fig_1.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
             frameon=False, handlelength=2.4, columnspacing=1.8, handletextpad=0.6)


# ======================================================================
# Relative L2 errors
# ======================================================================
def rel_l2(ref, pred):
    return (torch.linalg.norm(ref - pred, dim=0) / torch.linalg.norm(ref, dim=0)).detach().cpu().numpy()

order   = ["linspace", "random", "lhs", "sobol"]
rel     = np.stack([rel_l2(u_exac_tensor,     U_pred[k])  for k in order])   # (4 methods, 2)
rel_dot = np.stack([rel_l2(u_exac_dot_tensor, dU_pred[k]) for k in order])

print("Relative L2  [h, alpha]       :", dict(zip(order, rel.tolist())))
print("Relative L2  [hdot, alphadot] :", dict(zip(order, rel_dot.tolist())))

groups = [r'$h$', r'$\alpha$', r'$\dot{h}$', r'$\dot{\alpha}$']
data = np.array([rel[:, 0], rel[:, 1], rel_dot[:, 0], rel_dot[:, 1]])   # (4 groups, 4 methods)


# ======================================================================
# Figure 6 : bar chart of relative L2 error
# ======================================================================
fig_6, ax_6 = plt.subplots(figsize=(4.5, 3.6))

x = np.arange(len(groups))
width = 0.18
for i, lab in enumerate(["UG", "UR", "LHS", "Sobol"]):
    ax_6.bar(x + (i - 1.5) * width, data[:, i], width, label=lab)

ax_6.set_ylabel(r'Relative $L_2$ Error')
ax_6.set_ylim([0, .006])
ax_6.set_xticks(x)
ax_6.set_xticklabels(groups)
ax_6.tick_params(direction="in", axis="y", which="both", right=True)
ax_6.grid(False, axis="x")
ax_6.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax_6.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
            handlelength=1.5, columnspacing=1.5)
fig_6.tight_layout()


# ======================================================================
# Save, then show
# ======================================================================
save(fig_1, "Figure_1")
save(fig_6, "Figure_6")
plt.show()