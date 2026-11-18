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
from matplotlib.patches import Patch
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

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
names  = ["UG", "UR", "LHS", "Sobol"]
order  = ["linspace", "random", "lhs", "sobol"]
marks  = ["o", "s", "^", "d"]


def tag(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold")


# ======================================================================
# Relative L2 errors (for the bar chart)
# ======================================================================
def rel_l2(ref, pred):
    return (torch.linalg.norm(ref - pred, dim=0) / torch.linalg.norm(ref, dim=0)).detach().cpu().numpy()

rel     = np.stack([rel_l2(u_exac_tensor,     U_pred[k])  for k in order])   # (4 methods, 2)
rel_dot = np.stack([rel_l2(u_exac_dot_tensor, dU_pred[k]) for k in order])

groups = [r'$h$', r'$\alpha$', r'$\dot{h}$', r'$\dot{\alpha}$']
data = np.array([rel[:, 0], rel[:, 1], rel_dot[:, 0], rel_dot[:, 1]])   # (4 groups, 4 methods)


# ======================================================================
# One figure, two subplots side by side
#   (a) distribution of training points   (b) relative L2 error bars
# ======================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.4))

# ---- (a) training points: density heat strip (top) + coloured barcode (bottom) ----
T_END, WIN = 2.0, 0.2                     # time span (s), sliding window (s)
t_grid = np.linspace(0, T_END, 801)

def local_density(tt, win=WIN):
    """points inside the window / ideal (uniform) count; window is clipped at the
    domain edges so that the first/last 0.1 s are not artificially 'empty'."""
    lo = np.clip(t_grid - win / 2, 0, T_END)
    hi = np.clip(t_grid + win / 2, 0, T_END)
    cnt = ((tt[None, :] >= lo[:, None]) & (tt[None, :] <= hi[:, None])).sum(axis=1)
    return cnt / (len(tt) * (hi - lo) / T_END)

dens = [local_density(n(T_data[k]).ravel()) for k in order]
vmax = max(d.max() for d in dens)
pitch = 1.7                               # vertical distance between methods

for i, key in enumerate(order):           # UG at the bottom, Sobol on top (as before)
    tt = n(T_data[key]).ravel()
    y0 = i * pitch
    im = ax_a.imshow(dens[i][None, :], cmap="YlGnBu", vmin=0, vmax=vmax,
                     aspect="auto", origin="lower", interpolation="bilinear",
                     extent=[0, T_END, y0 + 0.62, y0 + 1.22])      # density strip
    ax_a.vlines(tt, y0 - 0.05, y0 + 0.52, color=colors[i], lw=1.2)  # coloured barcode

ax_a.set_yticks([i * pitch + 0.58 for i in range(len(order))])
ax_a.set_yticklabels(names)
ax_a.tick_params(axis="y", length=0)
ax_a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_a.set_xlim([0, T_END])
ax_a.set_ylim([-0.2, (len(order) - 1) * pitch + 1.4])
ax_a.tick_params(direction="in", axis="x", which="both", top=True)
ax_a.xaxis.set_major_locator(MaxNLocator(5))
ax_a.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_a.grid(False, axis="y")

# colour bar (outside the axes, so panel (a) keeps its size)
cax = ax_a.inset_axes([1.03, 0, 0.04, 1])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"Local density / ideal")
cb.ax.tick_params(direction="in")
cb.ax.grid(False)

# ---- (b) bar chart ----
x = np.arange(len(groups))
width = 0.18
for i in range(4):
    ax_b.bar(x + (i - 1.5) * width, data[:, i], width, label=names[i], color=colors[i])

ax_b.set_ylabel(r'Relative $L_2$ Error')
ax_b.set_ylim([0, .006])
ax_b.set_xticks(x)
ax_b.set_xticklabels(groups)
ax_b.tick_params(direction="in", axis="y", which="both", right=True)
ax_b.grid(False, axis="x")
ax_b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

for ax, s in zip([ax_a, ax_b], ["(a)", "(b)"]):
    tag(ax, s)

fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)

# ---- one shared legend ----
# h, l = ax_b.get_legend_handles_labels()
# fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
#            frameon=False, handlelength=1.5, columnspacing=1.8, handletextpad=0.6)

# legend: solid coloured rectangles (one per method)
h = [Patch(facecolor=c, edgecolor=c) for c in colors]
fig.legend(h, names, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
           frameon=False, handlelength=2.0, handleheight=1.0,
           columnspacing=1.8, handletextpad=0.6)
# ======================================================================
# Save, then show
# ======================================================================
fig.savefig("figures/Figure_12.png", dpi=600, bbox_inches="tight")
fig.savefig("figures/Figure_12.pdf", bbox_inches="tight")
plt.show()
