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
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.linewidth": 1.5,          # ضخامت کادر (spines)
    "xtick.major.width": 1.2,       # ضخامت تیک‌های اصلی
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,       # ضخامت تیک‌های فرعی
    "ytick.minor.width": 0.8,
    "xtick.major.size": 5,          # طول تیک‌ها
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.grid": True,              # گرید برای همه
    "grid.color": "0.8",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.axisbelow": True,         # گرید زیر منحنی‌ها
})

LEG = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
           handlelength=2.0, columnspacing=1.2, handletextpad=0.5, ncol=2)


def finish(ax, nx=5, ny=5):
    ax.tick_params(direction="out", length=5, width=1.6)
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", direction="out", length=3, width=1)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)


def tag(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold")


# ---------------- panel builders (draw on a given axes) ----------------
def hist_panel(ax, ep_p, y_p, ep_d, y_d, ylab, ylim):
    ax.semilogy(ep_p, y_p, color="#0057B8", lw=1.0, ls="-",  label=r"$\mathrm{PINN}$")
    ax.semilogy(ep_d, y_d, color="#8B1E3F", lw=1.0, ls="--", label=r"$\mathrm{Fully\ Scaled\ PINN}$", alpha=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 60000)
    ax.set_ylim(*ylim)
    ax.tick_params(direction="out", length=5, width=1.6)
    #ax.legend(**LEG)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[0.5]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", direction="out", length=3, width=1)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)


# def phase_panel(ax, k, xlab, ylab, lw_num, alpha_d, zorder_d):
#     ax.plot(u_ex[:, k], du_ex[:, k], label=r"$\mathrm{Numerical}$", color="#1E90FF", lw=lw_num, zorder=0)
#     ax.plot(Up[:, k],   dUp[:, k],   label=r"$\mathrm{PINN}$", color="black", ls="--", lw=1.0, zorder=1)
#     ax.plot(Ud[:, k],   dUd[:, k],   label=r"$\mathrm{Fully\ Scaled\ PINN}$", color="#DC143C",
#             ls="-.", lw=1.0, zorder=zorder_d, alpha=alpha_d)
#     ax.set_xlabel(xlab)
#     ax.set_ylabel(ylab)
#     #ax.legend(**LEG)
#     finish(ax)

# def phase_panel(ax, k, xlab, ylab, lw_num, alpha_d, zorder_d):
#     # Numerical: thick line at the bottom (reference)
#     ax.plot(u_ex[:, k], du_ex[:, k], label=r"$\mathrm{Numerical}$",
#             color="#1E90FF", lw=2.0, zorder=0)
#     # PINN: thin black line + hollow squares
#     ax.plot(Up[:, k], dUp[:, k], label=r"$\mathrm{PINN}$",
#             color="black", ls=None, lw=1.5, zorder=2,
#             marker="s", ms=3.5, mfc="black", mec="black", mew=0.9,
#             markevery=(0, 20))
#     # Fully Scaled PINN: red line + hollow circles, shifted so markers don't overlap
#     ax.plot(Ud[:, k], dUd[:, k], label=r"$\mathrm{Fully\ Scaled\ PINN}$",
#             color="#DC143C", ls=None,lw=1.0, zorder=3,
#             marker="o", ms=4.5, mfc="white", mec="#DC143C", mew=0.9,
#             markevery=(5, 30))
#     ax.set_xlabel(xlab)
#     ax.set_ylabel(ylab)
#     finish(ax)
def phase_panel(ax, k, xlab, ylab, lw_num, alpha_d, zorder_d):
    # Numerical: line only
    ax.plot(u_ex[:, k], du_ex[:, k], label=r"$\mathrm{Numerical}$",
            color="#1E90FF", lw=2.0, zorder=0)
    # PINN: markers only
    ax.plot(Up[:, k], dUp[:, k], label=r"$\mathrm{PINN}$",
            color="black", ls="none", zorder=2,
            marker="s", ms=3.5, mfc="black", mec="black", mew=0.9,
            markevery=(0, 20))
    # Fully Scaled PINN: markers only
    ax.plot(Ud[:, k], dUd[:, k], label=r"$\mathrm{Fully\ Scaled\ PINN}$",
            color="#DC143C", ls="none", zorder=3,
            marker="o", ms=4.5, mfc="white", mec="#DC143C", mew=0.9,
            markevery=(5, 30))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    finish(ax)

def time_panel(ax, k, Yex, Yp, Yd, ylab, sc_t, sc_y, sc_label, sc_s):
    ax.plot(t_ex, Yex[:, k], label=r"$\mathrm{Numerical}$", color="#1E90FF", lw=1.5, zorder=0)
    ax.plot(t_te, Yp[:, k],  label=r"$\mathrm{PINN}$", color="black", ls="--", lw=1.0, zorder=1)
    ax.plot(t_te, Yd[:, k],  label=r"$\mathrm{Fully\ Scaled\ PINN}$", color="#DC143C", ls="-.", lw=1.0, zorder=2)
    ax.scatter(sc_t, sc_y[:, k], label=sc_label, s=sc_s, marker="o", color="#ff7f0e", alpha=1, zorder=3)
    ax.set_xlim([-.05, 5.05])
    ax.set_xlabel(r"$t\,(\mathrm{s})$")
    ax.set_ylabel(ylab)
    ax.legend(**LEG)
    finish(ax, nx=6, ny=5)

# ======================================================================
# Shared legend helper (one legend on top of a whole figure)
# ======================================================================
def shared_legend(fig, ax, ncol, y=0.995):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False,
               handlelength=2.0, columnspacing=1.8, handletextpad=0.5)


# ======================================================================
# Figure 5 : loss and relative L2 error  (1 x 2)
# ======================================================================
fig_5, axs5 = plt.subplots(1, 2, figsize=(7.0, 4))

hist_panel(axs5[0], epoch_history_p, loss_history_p, epoch_history_d, loss_history_d,
           "Loss", (1e-5, 1e1))
hist_panel(axs5[1], epoch_history_p, l2_history_p, epoch_history_d, l2_history_d,
           r"Relative $L_2$ Error", (1e-4, 1e0))

for ax, s in zip(axs5, ["(a)", "(b)"]):
    tag(ax, s)
fig_5.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)       # جا برای لجند بالا
shared_legend(fig_5, axs5[0], ncol=2)


# ======================================================================
# Figure 6 : phase planes  (1 x 2)
# ======================================================================
fig_6, axs6 = plt.subplots(1, 2, figsize=(7.0, 4))

phase_panel(axs6[0], 0, r"$x_1$", r"$\dot{x_1}$", 2.5, 0.5, 2)
phase_panel(axs6[1], 1, r"$x_2$", r"$\dot{x_2}$", 2.0, 1.0, 1)

for ax, s in zip(axs6, ["(a)", "(b)"]):
    tag(ax, s)
fig_6.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)       # جا برای لجند بالا
shared_legend(fig_6, axs6[0], ncol=3)


# ======================================================================
# Save
# ======================================================================
for name, f in [("Figure_3", fig_5), ("Figure_4", fig_6)]:
    f.savefig(f"figures-m01/{name}.png", dpi=600, bbox_inches="tight")
    f.savefig(f"figures-m01/{name}.pdf", bbox_inches="tight")

plt.show()