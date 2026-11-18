# ======================================================================
# 1) Import the training module FIRST (it may wipe the namespace)
# ======================================================================
import sys, os
sys.path.append(os.path.abspath("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data"))
from Pitch_Plunge_2D_U_10_pinns_noisy_std_1_100 import PINNs_MLPs
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

DIR = "../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data"


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
    print(f"loaded {len(ws)} items from {path}")
    return ws


# ======================================================================
# Load all noise cases  (order = sigma 0, 1, 5, 10, 15, 20 %)
# ======================================================================
cases = ["0_100", "1_100", "5_100", "1_10", "15_100", "2_10"]

U_pred, dU_pred = {}, {}
scaler = None

for c in cases:
    ws = load_ws(f"{DIR}/pitch_plunge_2D_U_10_pinns_noisy_std_{c}.pth")

    # common tensors (same in every run; the last workspace is kept, as before)
    t_exac_tensor     = ws["t_exac_tensor"]
    u_exac_tensor     = ws["u_exac_tensor"]
    u_exac_dot_tensor = ws["u_exac_dot_tensor"]
    t_test_tensor     = ws["t_test_tensor"]

    if scaler is None:
        scaler = mmsptx(t_range=(-1, 1), x_range=(-1, 1), device=device).fit(
            t_exac_tensor, u_exac_tensor)

    model = PINNs_MLPs([1, 128, 128, 2]).to(device)
    model.load_state_dict(torch.load(f"{DIR}/pitch_plunge_2D_U_10_pinns_noisy_std_{c}_network.pth",
                                     map_location=device, weights_only=True))
    model.eval()

    t_s = ws["t_test_scaled_tensor"].detach().clone().requires_grad_(True)
    u_s = model(t_s)                                         # (N,2)
    du_s = torch.zeros_like(u_s)
    for k in range(2):
        du_s[:, k:k+1] = torch.autograd.grad(
            u_s[:, k:k+1], t_s,
            grad_outputs=torch.ones_like(u_s[:, k:k+1]),
            create_graph=True)[0]

    U_pred[c]  = scaler.inverse_transformX(u_s).detach()
    dU_pred[c] = scaler.scale_first_derivative_inverse_transform(du_s).detach()

n = lambda x: x.detach().cpu().numpy()
t_te = n(t_test_tensor)





def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")




# ======================================================================
# Relative L2 errors vs noise level
# ======================================================================
def rel_l2(ref, pred):
    return (torch.linalg.norm(ref - pred, dim=0) / torch.linalg.norm(ref, dim=0)).detach().cpu().numpy()

rel     = np.stack([rel_l2(u_exac_tensor,     U_pred[c])  for c in cases])   # (6, 2)
rel_dot = np.stack([rel_l2(u_exac_dot_tensor, dU_pred[c]) for c in cases])

rel_l2_h,     rel_l2_alpha     = rel[:, 0],     rel[:, 1]
rel_l2_h_dot, rel_l2_alpha_dot = rel_dot[:, 0], rel_dot[:, 1]

noise = np.array([0, 1, 5, 10, 15, 20])

# ======================================================================
# Journal-accurate setup
# ======================================================================
# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "mathtext.fontset": "custom",
#     "mathtext.rm": "Times New Roman",
#     "mathtext.it": "Times New Roman:italic",
#     "mathtext.bf": "Times New Roman:bold",
#     "axes.labelsize": 10,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8,
#     "axes.labelcolor": "black",
#     "xtick.color": "black",
#     "ytick.color": "black",
# })
# plt.close("all")
# ======================================================================
# Figure 2a  (final, designed from scratch)
# ======================================================================
# ======================================================================
# Figure 2a  (v3: clean style + relative deviation from mean, %)
# ======================================================================
# ======================================================================
# Figure 2a  (final: user palette + line styles + Delta from mean)
# ======================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.major.size": 5,    "ytick.major.size": 5,
    "xtick.minor.size": 2.8,  "ytick.minor.size": 2.8,
    "axes.axisbelow": True,
})

# ---------------- data ----------------
x      = np.asarray(noise, dtype=float)
data   = [np.asarray(d, dtype=float) for d in
          (rel_l2_h, rel_l2_alpha, rel_l2_h_dot, rel_l2_alpha_dot)]
labels = [r"$h$", r"$\alpha$", r"$\dot{h}$", r"$\dot{\alpha}$"]
mean   = np.mean(data, axis=0)

# ---------------- style ----------------
colors = ["#1B7F3B","#1E90FF","#000000",  "#DC143C", ]   #E69F00
lstyle = ["-.", "-", ":", "--"]
marker = ["o", "s", "^", "D"]
lwidth = [1.6, 2.5, 2.5, 2.5]
msize  = [9.0, 8.0, 7.0, 3.5]                  # concentric: big -> small
mew    = [1.2, 1.2, 1.2, 1.2]
mfc    = ["white", "none", "none", colors[3]]   # last one filled
zord   = [3, 4, 5, 6]

# ---------------- figure ----------------
fig_2a = plt.figure(figsize=(6.27, 5))
gs = fig_2a.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.0,
                         left=0.15, right=0.97, bottom=0.10, top=0.91)
ax  = fig_2a.add_subplot(gs[0])
axd = fig_2a.add_subplot(gs[1], sharex=ax)

# ---------------- main panel ----------------
for i in range(4):
    ax.plot(x, data[i], color=colors[i], lw=lwidth[i], ls=lstyle[i],
            marker=marker[i], ms=msize[i], mfc=mfc[i], mec=colors[i],
            mew=mew[i], zorder=zord[i])

ymax = max(d.max() for d in data)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 1.10 * ymax)
ax.set_ylabel(r"Relative $L_2$ error")
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(True, axis="both", which="major", color="0.90", lw=1.5,ls=":")
ax.tick_params(which="both", labelbottom=False, bottom=False)
ax.tick_params(which="both", top=False)          # 
ax.spines["bottom"].set_visible(False)           # مرز مشترک یکدست
# ---------------- deviation panel (each curve minus the mean) ----------------
axd.axhline(0, color="0.5", lw=1.0, zorder=1)
for i in range(4):
    axd.plot(x, data[i] - mean, color=colors[i], lw=lwidth[i], ls=lstyle[i],
             marker="none", ms=0.6 * msize[i], mfc=mfc[i], mec=colors[i],
             mew=1.0, zorder=zord[i])
lim = 1.25 * np.max(np.abs(np.array(data) - mean))
axd.set_ylim(-lim, lim)
axd.set_ylabel(r"$\Delta$ from mean")
axd.set_xlabel(r"Noise level, $p$ (% of signal std)")
axd.set_xlabel(r'Noise level $p$ (%)')
axd.xaxis.set_major_locator(MultipleLocator(5))
axd.xaxis.set_minor_locator(MultipleLocator(1))
axd.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
axd.yaxis.set_minor_locator(AutoMinorLocator(2))
axd.tick_params(which="both", top=False)
axd.grid(True, axis="both", which="major", color="0.90", lw=1.5,ls=":")

fig_2a.align_ylabels([ax, axd])

# ---------------- legend ----------------
handles = [Line2D([0], [0], color=colors[i], lw=lwidth[i], ls=lstyle[i],
                  marker=marker[i], ms=0.8 * msize[i], mfc=mfc[i],
                  mec=colors[i], mew=mew[i]) for i in range(4)]
fig_2a.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.995),
              ncol=4, frameon=False, handlelength=3.0, columnspacing=2.0,
              handletextpad=0.6)

# ======================================================================
# Save, then show
# ======================================================================
for name, f in [("Ex2_fig_2a", fig_2a)]:
    save(f, name)

plt.show()