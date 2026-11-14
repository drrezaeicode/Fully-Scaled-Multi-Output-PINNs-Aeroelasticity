# ======================================================================
# 1) Import the training module FIRST (it may wipe the namespace)
# ======================================================================
import sys, os
sys.path.append(os.path.abspath("../mass-spring-pinns-train-classic-m01"))
from mass_spring_pinns_train_classic_m01 import PINNs_MLPs
_PINN = PINNs_MLPs                      # safe handle

# ======================================================================
# 2) Re-import everything (safe even if the namespace was wiped)
# ======================================================================
import sys, os, gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, LogLocator, NullFormatter
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx   # My Library

PINNs_MLPs = _PINN
plt.close("all")
gc.collect()
device = "cpu"
os.makedirs("figures-m01", exist_ok=True)


# ======================================================================
# Helper: load a saved workspace into a dict (nothing touches globals)
#   - closes broken matplotlib figures created by unpickling
#   - drops matplotlib objects (fig, axs, ...) from the dict
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
# Classic PINN
# ======================================================================
ws_p = load_ws("../mass-spring-pinns-train-classic-m01/mass-spring-pinns-train-classic-m01.pth")

t_exac_tensor     = ws_p["t_exac_tensor"]
u_exac_tensor     = ws_p["u_exac_tensor"]
u_exac_dot_tensor = ws_p["u_exac_dot_tensor"]
t_test_tensor     = ws_p["t_test_tensor"]
t_data_tensor     = ws_p["t_data_tensor"]
u_data_tensor     = ws_p["u_data_tensor"]
t_ini_tensor      = ws_p["t_ini_tensor"]
du_ini_tensor     = ws_p["du_ini_tensor"]

loss_history_p  = list(ws_p["loss_history"])
l2_history_p    = list(ws_p["l2_history"])
epoch_history_p = list(ws_p["epoch_history"])

model = PINNs_MLPs([1, 128, 128, 2]).to(device)
model.load_state_dict(torch.load(
    "../mass-spring-pinns-train-classic-m01/mass-spring-pinns-train-classic-m01_network.pth",
    map_location=device, weights_only=True))
model.eval()

t_test_tensor_p = t_test_tensor.detach().clone().requires_grad_(True)
u_pred_tensor_p = model(t_test_tensor_p)                       # (N,2)
u_pred_dot_tensor_p = torch.zeros_like(u_pred_tensor_p)
for k in range(2):
    u_pred_dot_tensor_p[:, k:k+1] = torch.autograd.grad(
        u_pred_tensor_p[:, k:k+1], t_test_tensor_p,
        grad_outputs=torch.ones_like(u_pred_tensor_p[:, k:k+1]),
        create_graph=True)[0]


# ======================================================================
# Fully Scaled PINN
# ======================================================================
ws_d = load_ws("../mass-spring-pinns-train-scaled-m01/mass-spring-pinns-train-scaled-m01.pth")

loss_history_d  = list(ws_d["loss_history"])
l2_history_d    = list(ws_d["l2_history"])
epoch_history_d = list(ws_d["epoch_history"])

model_d = PINNs_MLPs([1, 128, 128, 2]).to(device)
model_d.load_state_dict(torch.load(
    "../mass-spring-pinns-train-scaled-m01/mass-spring-pinns-train-scaled-m01_network.pth",
    map_location=device, weights_only=True))
model_d.eval()

scaler = mmsptx(t_range=(-1, 1), x_range=(-1, 1), device=device).fit(t_exac_tensor, u_exac_tensor)

t_test_scaled_tensor_d = ws_d["t_test_scaled_tensor"].detach().clone().requires_grad_(True)
u_pred_scaled_tensor_d = model_d(t_test_scaled_tensor_d)       # (N,2)
u_pred_dot_scaled_tensor_d = torch.zeros_like(u_pred_scaled_tensor_d)
for k in range(2):
    u_pred_dot_scaled_tensor_d[:, k:k+1] = torch.autograd.grad(
        u_pred_scaled_tensor_d[:, k:k+1], t_test_scaled_tensor_d,
        grad_outputs=torch.ones_like(u_pred_scaled_tensor_d[:, k:k+1]),
        create_graph=True)[0]
u_pred_tensor_d     = scaler.inverse_transformX(u_pred_scaled_tensor_d)
u_pred_dot_tensor_d = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_d)


# ======================================================================
# To numpy
# ======================================================================
n = lambda x: x.detach().cpu().numpy()

t_ex, u_ex, du_ex = n(t_exac_tensor), n(u_exac_tensor), n(u_exac_dot_tensor)
t_te              = n(t_test_tensor)
t_da, u_da        = n(t_data_tensor), n(u_data_tensor)
t_in, du_in       = n(t_ini_tensor), n(du_ini_tensor)
Up,  dUp          = n(u_pred_tensor_p), n(u_pred_dot_tensor_p)
Ud,  dUd          = n(u_pred_tensor_d), n(u_pred_dot_tensor_d)


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
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

LEG = dict(loc="lower center", bbox_to_anchor=(0.45, 1.02), frameon=False,
           handlelength=2.0, columnspacing=1.2, handletextpad=0.5)


def finish(ax, nx=5, ny=5):
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", direction="in", length=2, width=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


# ======================================================================
# Figures 1a-1d : time histories
# ======================================================================
def time_fig(k, Yex, Yp, Yd, ylab, sc_t, sc_y, sc_label, sc_s):
    fig = plt.figure(figsize=(3.35, 2.7))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.18, bottom=0.18)
    ax.plot(t_ex, Yex[:, k], label=r"$\mathrm{Numerical}$", color="#1E90FF", lw=1.5, zorder=0)
    ax.plot(t_te, Yp[:, k],  label=r"$\mathrm{PINN}$", color="black", ls="--", lw=1.0, zorder=1)
    ax.plot(t_te, Yd[:, k],  label=r"$\mathrm{Fully\ Scaled\ PINN}$", color="#DC143C", ls="-.", lw=1.0, zorder=2)
    ax.scatter(sc_t, sc_y[:, k], label=sc_label, s=sc_s, marker="o", color="#ff7f0e", alpha=1, zorder=3)
    ax.set_xlim([-.05, 5.05])
    ax.set_xlabel(r"$t\,(\mathrm{s})$")
    ax.set_ylabel(ylab)
    ax.legend(ncol=4, **LEG)
    finish(ax, nx=6, ny=5)
    fig.tight_layout()
    return fig

fig_1a = time_fig(0, u_ex,  Up,  Ud,  r"$x_1$",       t_da, u_da,  r"$\mathrm{Observed}$", 6)
fig_1b = time_fig(1, u_ex,  Up,  Ud,  r"$x_2$",       t_da, u_da,  r"$\mathrm{Observed}$", 6)
fig_1c = time_fig(0, du_ex, dUp, dUd, r"$\dot{x_1}$", t_in, du_in, r"$\mathrm{Initial}$",  10)
fig_1d = time_fig(1, du_ex, dUp, dUd, r"$\dot{x_2}$", t_in, du_in, r"$\mathrm{Initial}$",  10)


# ======================================================================
# Figures 2a, 2b : phase planes
# ======================================================================
def phase_fig(k, xlab, ylab, lw_num, alpha_d, zorder_d):
    fig = plt.figure(figsize=(3.35, 3.35))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.18, bottom=0.18)
    ax.plot(u_ex[:, k], du_ex[:, k], label=r"$\mathrm{Numerical}$", color="#1E90FF", lw=lw_num, zorder=0)
    ax.plot(Up[:, k],   dUp[:, k],   label=r"$\mathrm{PINN}$", color="black", ls="--", lw=1.0, zorder=1)
    ax.plot(Ud[:, k],   dUd[:, k],   label=r"$\mathrm{Fully\ Scaled\ PINN}$", color="#DC143C",
            ls="-.", lw=1.0, zorder=zorder_d, alpha=alpha_d)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(ncol=4, **LEG)
    finish(ax)
    fig.tight_layout()
    return fig

fig_2a = phase_fig(0, r"$x_1$", r"$\dot{x_1}$", 2.5, 0.5, 2)
fig_2b = phase_fig(1, r"$x_2$", r"$\dot{x_2}$", 2.0, 1.0, 1)


# ======================================================================
# Figures 3a, 3b : loss / L2 history
# ======================================================================
def hist_fig(ep_p, y_p, ep_d, y_d, ylab, ylim):
    fig = plt.figure(figsize=(3.35, 3.35))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.18, bottom=0.18)
    ax.semilogy(ep_p, y_p, color="#0057B8", lw=1.0, ls="-",  label=r"$\mathrm{PINN}$")
    ax.semilogy(ep_d, y_d, color="#8B1E3F", lw=1.0, ls="--", label=r"$\mathrm{Fully\ Scaled\ PINN}$", alpha=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 60000)
    ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.legend(ncol=2, **LEG)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[0.5]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", direction="in", length=2, width=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    fig.tight_layout()
    return fig

fig_3a = hist_fig(epoch_history_p, loss_history_p, epoch_history_d, loss_history_d, "Loss", (4e-4, 1e1))
fig_3b = hist_fig(epoch_history_p, l2_history_p,   epoch_history_d, l2_history_d,
                  r"Relative $L_2$ Error", (3e-3, 1e0))


# ======================================================================
# Figure 6br (reduced version: no ticks)
# ======================================================================
fig_2br = plt.figure(figsize=(4.0, 3.5))
ax_2br = fig_2br.add_subplot()
fig_2br.subplots_adjust(left=0.18, bottom=0.18)

ax_2br.plot(u_ex[:, 1], du_ex[:, 1], label=r"$\mathrm{Numerical}$",
            color="#3A3A3A", linewidth=3.0, zorder=0)
ax_2br.plot(Ud[:, 1], dUd[:, 1], label=r"$\mathrm{Fully\ Scaled\ PINN}$",
            color="red", linewidth=2.5, linestyle="--", zorder=1)

ax_2br.set_xlabel(r"$x_2$", fontsize=20)
ax_2br.set_ylabel(r"$\dot{x}_2$", fontsize=20)
ax_2br.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
              fontsize=12, handlelength=2.5, columnspacing=1.5, handletextpad=0.6)

ax_2br.set_xticks([])
ax_2br.set_yticks([])
ax_2br.minorticks_off()
ax_2br.tick_params(which="both", bottom=False, top=False, left=False, right=False)
for sp in ax_2br.spines.values():
    sp.set_linewidth(1.2)
fig_2br.tight_layout()


# ======================================================================
# Save (all figures are fresh, none comes from the pickled workspace)
# ======================================================================
for name, f in [("Ex2_fig_1a", fig_1a), ("Ex2_fig_1b", fig_1b),
                ("Ex2_fig_1c", fig_1c), ("Ex2_fig_1d", fig_1d),
                ("Figure_6a",  fig_2a), ("Figure_6b",  fig_2b),
                ("Figure_5a",  fig_3a), ("Figure_5b",  fig_3b),
                ("Figure_6br", fig_2br)]:
    f.savefig(f"figures-m01/{name}.png", dpi=600, bbox_inches="tight")
    f.savefig(f"figures-m01/{name}.pdf", bbox_inches="tight")

plt.show()