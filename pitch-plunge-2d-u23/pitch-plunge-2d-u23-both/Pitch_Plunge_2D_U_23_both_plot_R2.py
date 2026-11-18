# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:42:18 2026

@author: Mohsen
"""


from IPython import get_ipython
get_ipython().run_line_magic('clear','')
get_ipython().run_line_magic('reset' , '-f')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
device = "cpu"
import numpy as np

import matplotlib.pyplot as plt
plt.close("all")
import gc
gc.collect()

import sys
import os
sys.path.append(os.path.abspath("../pitch-plunge-2d-u23-pinns-train"))
from Pitch_Plunge_2D_U_23_pinns_train import PINNs_MLPs
from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx  # My Library

# ======================================================================
# Load PINN model + prediction
# ======================================================================
loaded_workspace = load_globals("../pitch-plunge-2d-u23-pinns-train/pitch_plunge_2D_U_23_pinns_train.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device)
model.load_state_dict(torch.load("../pitch-plunge-2d-u23-pinns-train/pitch_plunge_2D_U_23_pinns_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)

t_test_scaled_tensor = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor = model(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor = torch.zeros_like(u_pred_scaled_tensor)
u_pred_dot_scaled_tensor[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor[:, 0:1], t_test_scaled_tensor, grad_outputs=torch.ones_like(u_pred_scaled_tensor[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor[:, 1:2], t_test_scaled_tensor, grad_outputs=torch.ones_like(u_pred_scaled_tensor[:, 0:1]), create_graph=True)[0]
u_pred_tensor = scaler.inverse_transformX(u_pred_scaled_tensor)
u_pred_dot_tensor = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor)

# ======================================================================
# Load DDNN model + prediction
# ======================================================================
loaded_workspace = load_globals("../pitch-plunge-2d-u23-data-train/pitch_plunge_2D_U_23_data_train.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device)
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u23-data-train/pitch_plunge_2D_U_23_data_network.pth", map_location=device, weights_only=True))
model_d.eval()

scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_d = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_d = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_d = torch.zeros_like(u_pred_scaled_tensor_d)
u_pred_dot_scaled_tensor_d[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_d[:, 0:1], t_test_scaled_tensor, grad_outputs=torch.ones_like(u_pred_scaled_tensor_d[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_d[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_d[:, 1:2], t_test_scaled_tensor, grad_outputs=torch.ones_like(u_pred_scaled_tensor_d[:, 0:1]), create_graph=True)[0]
u_pred_tensor_d = scaler.inverse_transformX(u_pred_scaled_tensor_d)
u_pred_dot_tensor_d = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_d)

# ======================================================================
# Convert to numpy + error computation
# ======================================================================
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, MultipleLocator
from matplotlib.lines import Line2D

np_ = lambda x: x.detach().cpu().numpy()

t_ex   = np_(t_exac_tensor).ravel()
t_te   = np_(t_test_tensor).ravel()
t_da   = np_(t_data_tensor).ravel()

U_ex   = np_(u_exac_tensor)          # (Nex,2)  [h, alpha]
dU_ex  = np_(u_exac_dot_tensor)      # (Nex,2)  [hdot, alphadot]
U_p    = np_(u_pred_tensor)          # PINN
dU_p   = np_(u_pred_dot_tensor)
U_d    = np_(u_pred_tensor_d)        # DDNN
dU_d   = np_(u_pred_dot_tensor_d)
U_da   = np_(u_data_tensor)

def interp_ex(Y):
    return np.stack([np.interp(t_te, t_ex, Y[:, k]) for k in range(Y.shape[1])], axis=1)

U_ex_i  = interp_ex(U_ex)
dU_ex_i = interp_ex(dU_ex)

# # error = prediction - numerical
# E_p,  E_d  = U_p  - U_ex_i,  U_d  - U_ex_i       # error in h, alpha
# dE_p, dE_d = dU_p - dU_ex_i, dU_d - dU_ex_i      # error in hdot, alphadot
# error = prediction - numerical()/NUMERICAL


scale  = np.max(np.abs(U_ex),  axis=0)     # برای h و alpha
dscale = np.max(np.abs(dU_ex), axis=0)     # برای hdot و alphadot

E_p,  E_d  = 100*(U_p  - U_ex_i )/scale,  100*(U_d  - U_ex_i )/scale
dE_p, dE_d = 100*(dU_p - dU_ex_i)/dscale, 100*(dU_d - dU_ex_i)/dscale

# L2 relative error (single number per quantity)
def rel_L2(p, r):
    return np.linalg.norm(p - r, axis=0) / np.linalg.norm(r, axis=0)

print("L2 rel. error  [h, alpha]       PINN:", rel_L2(U_p,  U_ex_i),  " DDNN:", rel_L2(U_d,  U_ex_i))
print("L2 rel. error  [hdot, alphadot] PINN:", rel_L2(dU_p, dU_ex_i), " DDNN:", rel_L2(dU_d, dU_ex_i))



# ======================================================================
# Journal-accurate setup
# ======================================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 9,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

# ---- journal colors ----
# C_NUM  = "#1E90FF"#"#1F3A93"   # Numerical  (navy)
# C_PINN = "#000000"   # Observed (black, hollow marker)
# C_DDNN = "#DC143C"#"#B2182B"   # Fully Scaled PINN (red)
# C_OBS  = "#E69F00"#"#FF00FF"#"#FFD700"  # DDNN (green)



C_NUM = "#B8BEC6"   # reference  : light grey (thick, underneath)
C_PINN = "#1F18A8"#"#0072B2"   # fully scaled PINN : blue
C_DDNN = "#DC143C"#"#D55E00"   # DDNN       : vermillion
C_OBS  = "#000000"   # observations : black hollow circles

'''

ax_1b.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor_d[:, 1].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=2)
ax_1b.scatter(t_data_tensor.detach().cpu().numpy(),u_data_tensor[:, 1].detach().cpu().numpy(), label=r"$\mathrm{Observed}$"    ,s=2,marker="o",color="#FFD700",alpha=1,zorder=3)
'''
# ---- line styles (thick numerical at the bottom, PINN on top) ----
# ST_NUM  = dict(color=C_NUM,  lw=1.5, ls="-",            zorder=1)
# ST_DDNN = dict(color=C_DDNN, lw=1.1, ls="-.",           zorder=2)
# ST_PINN = dict(color=C_PINN, lw=1.5, ls=(0, (4, 2)),    zorder=3)

ST_NUM  = dict(color=C_NUM,  lw=3.5, ls="-",           zorder=1, solid_capstyle="round")
ST_DDNN = dict(color=C_DDNN, lw=1.2, ls=(0, (5, 1.5, 1, 1.5)), zorder=2)
ST_PINN = dict(color=C_PINN, lw=1.4, ls="--", zorder=3)#(0, (4, 2.5))

# LABELS = [r"$\mathrm{Numerical}$", r"$\mathrm{Fully\ Scaled\ PINN}$",
#           r"$\mathrm{DDNN}$", r"$\mathrm{Observed}$"]
LABELS = [r"$\mathrm{Reference\ (numerical)}$",
          r"$\mathrm{Fully\ Scaled\ PINN}$",
          r"$\mathrm{DDNN}$",
          r"$\mathrm{Observations}\ (h,\,\alpha)$"]
def style(ax, nticks=5):
    ax.tick_params(which="major", direction="in", length=4, width=0.8, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=2, width=0.5, top=True, right=True)
    ax.xaxis.set_major_locator(MaxNLocator(nticks))
    ax.yaxis.set_major_locator(MaxNLocator(nticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color="0.8", lw=0.5, ls="-", zorder=0)   # ← این خط اضافه شد
    ax.set_axisbelow(True)                                                  # ← گرید زیر منحنی‌ها
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)

def obs_scatter(ax, x, y):
    ax.scatter(x, y, s=20, marker="o", facecolors="white", edgecolors=C_OBS,
               linewidths=1.2, zorder=5)

legend_handles = [
    Line2D([0], [0], color=C_NUM,  lw=3.5, ls="-"),
    Line2D([0], [0], color=C_PINN, lw=1.4, ls=(0, (4, 2))),
    Line2D([0], [0], color=C_DDNN, lw=1.1, ls="-."),
    Line2D([0], [0], color=C_OBS,  ls="none", marker="o", ms=4.5, mfc="white", mew=1.2),
]

# ======================================================================
# Figure 9 (a-f): single figure
#   top    : 4 x 1 time panels (each with its error panel)
#   bottom : 1 x 2 phase planes
# ======================================================================
LABEL_PANELS = True
XLABEL_ONLY_LAST = True      # "t (s)" only under the last time panel

fig = plt.figure(figsize=(6.27, 8.5))     # ارتفاع بیشتر چون ردیف فاز‌پلین اضافه شده
root = fig.add_gridspec(2, 1, left=0.12, right=0.98, bottom=0.05, top=0.965,
                        height_ratios=[4*2, 3.5], hspace=0.15)   # ### نسبت ردیف بالا به پایین

# --- ردیف بالا: 4 پنل زمانی ---
outer = root[0].subgridspec(4, 1, hspace=0.2)   ###فاصله بین  چهار پلات اول
all_axes = []

def time_panel(spec, k, Y_ex, Y_p, Y_d, E_pinn, E_ddnn, ylab, elab,
               data=None, tag="", last=False):
    sub = spec.subgridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.10)###
    ax  = fig.add_subplot(sub[0])
    axe = fig.add_subplot(sub[1], sharex=ax)

    ax.plot(t_ex, Y_ex[:, k], **ST_NUM)
    ax.plot(t_te, Y_d[:, k],  **ST_DDNN)
    ax.plot(t_te, Y_p[:, k],  **ST_PINN)
    if data is not None:
        obs_scatter(ax, t_da, data[:, k])
    ax.set_xlim([-.01, 2.01])
    ax.set_ylabel(ylab)
    ax.tick_params(labelbottom=False)
    style(ax)
    
    ax.yaxis.set_major_locator(MaxNLocator(4))
    #ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    if LABEL_PANELS:
        ax.text(-0.05, 1.03, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, fontweight="bold")

    axe.axhline(0, color=C_NUM, lw=3.5, zorder=0)
    axe.plot(t_te, E_ddnn[:, k], **{**ST_DDNN, "lw": 1.1})
    axe.plot(t_te, E_pinn[:, k], **{**ST_PINN, "lw": 1.4})
    axe.set_ylabel(elab)
    style(axe, nticks=3)
    axe.xaxis.set_major_locator(MultipleLocator(0.4))
    axe.xaxis.set_minor_locator(MultipleLocator(0.2))
    if last or not XLABEL_ONLY_LAST:
        axe.set_xlabel(r"$t\,(\mathrm{s})$", labelpad=-1)
    all_axes.extend([ax, axe])

time_panel(outer[0], 0, U_ex,  U_p,  U_d,  E_p,  E_d,  r"$h$",
           r"$\Delta h\ (\%)$",           data=U_da, tag="(a)")
time_panel(outer[1], 1, U_ex,  U_p,  U_d,  E_p,  E_d,  r"$\alpha$",
           r"$\Delta\alpha\ (\%)$",       data=U_da, tag="(b)")
time_panel(outer[2], 0, dU_ex, dU_p, dU_d, dE_p, dE_d, r"$\dot{h}$",
           r"$\Delta\dot{h}\ (\%)$",      tag="(c)")
time_panel(outer[3], 1, dU_ex, dU_p, dU_d, dE_p, dE_d, r"$\dot{\alpha}$",
           r"$\Delta\dot{\alpha}\ (\%)$", tag="(d)", last=True)

fig.align_ylabels(all_axes)

# --- ردیف پایین: 2 فاز‌پلین کنار هم ---
bottom = root[1].subgridspec(1, 2, wspace=0.35)   ### فاصله افقی دو فاز‌پلین
axp1 = fig.add_subplot(bottom[0])
axp2 = fig.add_subplot(bottom[1])

for axp, k, xlab, ylab, tag in [(axp1, 0, r"$h$",      r"$\dot{h}$",      "(e)"),
                                (axp2, 1, r"$\alpha$", r"$\dot{\alpha}$", "(f)")]:
    axp.plot(U_ex[:, k], dU_ex[:, k], **ST_NUM)
    axp.plot(U_d[:, k],  dU_d[:, k],  **ST_DDNN)
    axp.plot(U_p[:, k],  dU_p[:, k],  **{**ST_PINN,"lw":1.1})
    axp.plot(U_ex[0, k], dU_ex[0, k], marker="o", ms=5, mfc="black", mec="white",
         mew=0.8, ls="none", zorder=6)
    axp.set_xlabel(xlab)
    axp.set_ylabel(ylab)
    style(axp)
    axp.text(-0.10, 1.04, tag, transform=axp.transAxes, ha="right", va="bottom",
             fontsize=10, fontweight="bold")

# --- یک لجند مشترک بالای کل فیگور ---
fig.legend(legend_handles, LABELS, loc="upper center", bbox_to_anchor=(0.55, 0.999),
           ncol=4, frameon=False, handlelength=2.4, columnspacing=1.8, handletextpad=0.6)

plt.show()

# ======================================================================
# Save
# ======================================================================
fig.savefig("figures/Figure_10.png", dpi=600, bbox_inches="tight")
fig.savefig("figures/Figure_10.pdf", bbox_inches="tight")