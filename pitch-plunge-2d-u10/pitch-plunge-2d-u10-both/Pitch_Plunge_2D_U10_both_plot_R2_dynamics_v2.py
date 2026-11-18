# from IPython import get_ipython
# get_ipython().run_line_magic('clear','')
# get_ipython().run_line_magic('reset' , '-f')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
device = "cpu"
import numpy as np

import matplotlib.pyplot as plt
plt.close("all")
import gc
gc.collect()           # آزاد کردن حافظه اشغال شده توسط figure های قدیمی

import sys
import os
sys.path.append(os.path.abspath("../pitch-plunge-2d-u10-pinns-train"))
from Pitch_Plunge_2D_U_10_pinns_train import PINNs_MLPs  # اگر در ماژول جدا ذخیره شده
from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_train.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor=t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor = model(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor = torch.zeros_like(u_pred_scaled_tensor)
u_pred_dot_scaled_tensor[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor[:, 0:1], t_test_scaled_tensor, grad_outputs=torch.ones_like(u_pred_scaled_tensor[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor[:, 1:2], t_test_scaled_tensor, grad_outputs=torch.ones_like(u_pred_scaled_tensor[:, 0:1]), create_graph=True)[0]
u_pred_tensor = scaler.inverse_transformX(u_pred_scaled_tensor)
u_pred_dot_tensor = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor)




loaded_workspace = load_globals("../pitch-plunge-2d-u10-data-train/pitch_plunge_2D_U_10_data_train.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-data-train/pitch_plunge_2D_U_10_data_network.pth", map_location=device, weights_only=True))
model_d.eval()

scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
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
C_PINN = "#1F18A8" #"#0072B2"   # fully scaled PINN : blue
C_DDNN = "#DC143C"#"#D55E00"   # DDNN       : vermillion
C_OBS  = "#000000"   # observations : black hollow circles


'''

ax_1b.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor_d[:, 1].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=2)
ax_1b.scatter(t_data_tensor.detach().cpu().numpy(),u_data_tensor[:, 1].detach().cpu().numpy(), label=r"$\mathrm{Observed}$"    ,s=2,marker="o",color="#FFD700",alpha=1,zorder=3)
'''
# ---- line styles (thick numerical at the bottom, PINN on top) ----
# ST_NUM  = dict(color=C_NUM,  lw=2, ls="-",            zorder=1)
# ST_DDNN = dict(color=C_DDNN, lw=1.1, ls="-.",           zorder=2)
# ST_PINN = dict(color=C_PINN, lw=1.5, ls=(0, (4, 2)),    zorder=3)

# ST_NUM  = dict(color=C_NUM,  lw=2.5, ls="-",           zorder=1)
# ST_DDNN = dict(color=C_DDNN, lw=1.1, ls="-.",          zorder=2)
# ST_PINN = dict(color=C_PINN, lw=1.5, ls=(0, (4, 3)),   zorder=3)
ST_NUM  = dict(color=C_NUM,  lw=3.5, ls="-",           zorder=1, solid_capstyle="round")
ST_DDNN = dict(color=C_DDNN, lw=1.1, ls=(0, (5, 1.5, 1, 1.5)), zorder=2)
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

# # --- ردیف بالا: 4 پنل زمانی ---
outer = root[0].subgridspec(4, 1, hspace=0.2)   ###فاصله بین  چهار پلات اول
all_axes = []

def time_panel(spec, k, Y_ex, Y_p, Y_d, E_pinn, E_ddnn, ylab, elab,
               data=None, tag="", last=False):
    sub = spec.subgridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.1)###
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
    
    ax.yaxis.set_major_locator(MaxNLocator(3))
    #ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    if LABEL_PANELS:
        ax.text(-0.13, 0.95, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, fontweight="bold")

    axe.axhline(0, color=C_NUM, lw=3.5, zorder=0)
    axe.plot(t_te, E_ddnn[:, k],**ST_DDNN )#, "lw": 1.0}
    axe.plot(t_te, E_pinn[:, k],**ST_PINN )#**{, "lw": 1.0}
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

# # --- ردیف پایین: 2 فاز‌پلین کنار هم ---
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

# # # --- یک لجند مشترک بالای کل فیگور ---
fig.legend(legend_handles, LABELS, loc="upper center", bbox_to_anchor=(0.55, 0.999),
           ncol=4, frameon=False, handlelength=2.4, columnspacing=1.8, handletextpad=0.6)

plt.show()

# ======================================================================
# Save
# ======================================================================
fig.savefig("figures/Figure_09.png", dpi=600, bbox_inches="tight")
fig.savefig("figures/Figure_09.pdf", bbox_inches="tight")


# ======================================================================
# ======================================================================
#  ADDED BLOCK  --  Reviewer 1, Comment 5: dynamic validation
#  (frequency, growth/decay rate, peak amplitude, errors, LCO check)
#  Uses only variables already created above; no retraining needed.
# ======================================================================
# ======================================================================
from scipy.signal import find_peaks            # (kept for optional use)
import os, csv

CASE_TAG     = "U10"     # used in file names
U_VALUE      = 10        # flow speed of this case (m/s)
LOG_ENVELOPE = False     # semilog y-axis for the envelope panels
POST_FLUTTER = False     # True: also report the periodic-response (LCO) check
N_LAST       = 5           # number of last cycles used in the LCO check
LATE_FRAC    = 0.30        # last 30 % of the record = "late-time" window

os.makedirs("figures", exist_ok=True)
os.makedirs("tables",  exist_ok=True)

# ---- uniform time grid (needed for FFT / windows) ----------------------
tu = np.linspace(t_te[0], t_te[-1], len(t_te))
dt = tu[1] - tu[0]
T_tot = tu[-1] - tu[0]
u_  = lambda y: np.interp(tu, t_te, y)

STATES = [("h", 0), ("alpha", 1)]
SERIES = {                      # method -> (displacements, velocities), physical units
    "Reference":         (U_ex_i, dU_ex_i),
    "Fully scaled PINN": (U_p,    dU_p),
    "DDNN":              (U_d,    dU_d),
}

# ---- 1) dominant frequency --------------------------------------------
def freq_fft(y):
    """dominant frequency [Hz]: Hann window + zero padding + parabolic peak interpolation"""
    y0 = y - y.mean()
    n  = 2**int(np.ceil(np.log2(len(y0))) + 3)
    S  = np.abs(np.fft.rfft(y0*np.hanning(len(y0)), n)); f = np.fft.rfftfreq(n, dt)
    i  = int(np.argmax(S))
    if 0 < i < len(S)-1:
        a, b, c = np.log(S[i-1]+1e-30), np.log(S[i]+1e-30), np.log(S[i+1]+1e-30)
        return f[i] + 0.5*(a-c)/(a-2*b+c)*(f[1]-f[0])
    return f[i]

def freq_zero_cross(y):
    """cross-check [Hz]: mean period between upward zero crossings"""
    y0 = y - y.mean()
    k  = np.where((y0[:-1] < 0) & (y0[1:] >= 0))[0]
    if len(k) < 2: return np.nan
    tc = tu[k] - y0[k]*dt/(y0[k+1]-y0[k])
    return (len(tc)-1)/(tc[-1]-tc[0])

def spectrum(y):
    y0 = y - y.mean()
    n  = 2**int(np.ceil(np.log2(len(y0))) + 3)
    return np.fft.rfftfreq(n, dt), np.abs(np.fft.rfft(y0*np.hanning(len(y0)), n))

# ---- 2) amplitude envelope (one value per period of the reference) ----
def envelope(y, P):
    """max|y| in consecutive windows of length P  ->  (window-centre time, amplitude)"""
    nw = int(np.floor(T_tot/P))
    tt, aa = [], []
    for j in range(nw):
        m = (tu >= tu[0]+j*P) & (tu < tu[0]+(j+1)*P)
        if m.sum() > 2:
            tt.append(tu[0]+(j+0.5)*P); aa.append(np.max(np.abs(y[m])))
    return np.array(tt), np.array(aa)

def growth_rate(tt, aa):
    """sigma [1/s] of A ~ exp(sigma t): all windows if decaying;
       only the growth phase (up to 90 % of the maximum) if growing."""
    n = len(aa); q = max(n//4, 1)
    growing = aa[-q:].mean() > aa[:q].mean()
    if growing:
        ii  = int(np.where(aa >= 0.9*aa.max())[0][0])
        sel = slice(0, max(ii+1, 4))
    else:
        sel = slice(0, n)
    s = np.polyfit(tt[sel], np.log(np.maximum(aa[sel], 1e-30)), 1)[0]
    return s, ("growth" if growing else "decay")

# ---- compute the dynamic-metrics table --------------------------------
rows_dyn, env_store, spec_store = [], {}, {}
for sname, k in STATES:
    P_ref = 1.0/freq_fft(u_(SERIES["Reference"][0][:, k]))
    for meth, (Q, dQ) in SERIES.items():
        y   = u_(Q[:, k])
        f1  = freq_fft(y); f2 = freq_zero_cross(y)
        tt, aa = envelope(y, P_ref)
        sg, kind = growth_rate(tt, aa)
        zeta = -sg/np.sqrt(sg**2 + (2*np.pi*f1)**2)
        rows_dyn.append([U_VALUE, sname, meth, f1, f2, sg, zeta, kind, np.max(np.abs(y))])
        env_store[(sname, meth)]  = (tt, aa)
        spec_store[(sname, meth)] = spectrum(y)

# ---- LCO / periodic-response check ------------------------------------
rows_lco = []
for sname, k in STATES:
    for meth in SERIES:
        tt, aa = env_store[(sname, meth)]
        last   = aa[-N_LAST:]
        var    = 100*(last.max()-last.min())/last.mean()
        m      = tt >= tt[-1] - LATE_FRAC*T_tot
        s_late = np.polyfit(tt[m], np.log(np.maximum(aa[m], 1e-30)), 1)[0] if m.sum() >= 3 else np.nan
        rows_lco.append([U_VALUE, sname, meth, aa[-3:].mean(), var, s_late])

# ---- peak amplitude + prediction errors (all four states) -------------
def rel_pct(a, b): return 100*abs(a-b)/abs(b)
rows_err = []
names4 = [("h",0,U_ex_i,U_p,U_d,E_p,E_d), ("alpha",1,U_ex_i,U_p,U_d,E_p,E_d),
          ("hdot",0,dU_ex_i,dU_p,dU_d,dE_p,dE_d), ("alphadot",1,dU_ex_i,dU_p,dU_d,dE_p,dE_d)]
for nm, k, R, Pp, Dd, Ep, Ed in names4:
    pr, pp, pdn = np.max(np.abs(R[:,k])), np.max(np.abs(Pp[:,k])), np.max(np.abs(Dd[:,k]))
    rows_err.append([U_VALUE, nm, pr, pp, rel_pct(pp,pr), pdn, rel_pct(pdn,pr),
                     100*rel_L2(Pp,R)[k], 100*rel_L2(Dd,R)[k],
                     np.max(np.abs(Ep[:,k])), np.max(np.abs(Ed[:,k]))])

# ---- print + save -----------------------------------------------------
def save_csv(name, header, rows):
    with open(f"tables/{name}_{CASE_TAG}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)

h1 = ["U","state","method","f_FFT[Hz]","f_zero-cross[Hz]","sigma[1/s]","zeta_eff","phase","max|q|"]
h2 = ["U","state","method","last3-window amplitude","variation of last %d cycles [%%]"%N_LAST,"late-time sigma[1/s]"]
h3 = ["U","state","max|ref|","max|PINN|","diff PINN[%]","max|DDNN|","diff DDNN[%]",
      "relL2 PINN[%]","relL2 DDNN[%]","max err PINN[%]","max err DDNN[%]"]
save_csv("TableR15_dynamics", h1, rows_dyn)
if POST_FLUTTER: save_csv("TableR15_LCO", h2, rows_lco)
save_csv("TableR15_peak_error", h3, rows_err)

print("\n=== (1) frequency / growth-decay rate  (U = %s) ===" % U_VALUE)
for r in rows_dyn:
    print(f"{r[1]:6s} {r[2]:18s} f_FFT={r[3]:7.3f} Hz  f_zc={r[4]:7.3f} Hz  sigma={r[5]:8.3f} 1/s  zeta_eff={r[6]:7.4f} ({r[7]})  max|q|={r[8]:.4g}")
print("\n--- relative difference to the reference [%] ---")
for sname, _ in STATES:
    ref = [r for r in rows_dyn if r[1]==sname and r[2]=="Reference"][0]
    for meth in ("Fully scaled PINN","DDNN"):
        r = [x for x in rows_dyn if x[1]==sname and x[2]==meth][0]
        print(f"{sname:6s} {meth:18s} d_f={rel_pct(r[3],ref[3]):6.2f}  d_sigma={rel_pct(r[5],ref[5]) if abs(ref[5])>1e-9 else np.nan:7.2f}  d_peak={rel_pct(r[8],ref[8]):6.2f}")

if POST_FLUTTER:
  print("\n=== (2) periodic-response (LCO) check, last %d cycles ===" % N_LAST)
  for r in rows_lco:
      print(f"{r[1]:6s} {r[2]:18s} final amp={r[3]:.4g}  variation of last {N_LAST} cycles={r[4]:6.2f} %  late-time sigma={r[5]:8.3f} 1/s")
  for sname, _ in STATES:
    r = [x for x in rows_lco if x[1]==sname and x[2]=="Reference"][0]
    print(f"[hint] reference {sname}: ", "amplitude saturated -> periodic (LCO)" if (r[4] < 5 and abs(r[5]) < 0.5) else "amplitude NOT saturated within the record -> do not claim LCO")

print("\n=== (3) peak amplitude and errors (physical units) ===")
for r in rows_err:
    print(f"{r[1]:9s} max|ref|={r[2]:.4g}  PINN {r[3]:.4g} ({r[4]:.2f}%)  DDNN {r[5]:.4g} ({r[6]:.2f}%)  | relL2 PINN {r[7]:.2f}% DDNN {r[8]:.2f}% | max err PINN {r[9]:.1f}% DDNN {r[10]:.1f}%")

# ---- figure 2: amplitude envelope (a,b) + spectrum (c,d)  -> 4 x 1 stack --
#      (a,b) share the t-axis, (c,d) share the f-axis
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

def envelope_smooth(y, P, n_pts=1000):
    """smooth envelope for PLOTTING only (tables above still use `envelope`):
       peaks of |y| (one per half period) joined by a monotone cubic (PCHIP) curve,
       so the line passes through the real peaks without kinks."""
    a = np.abs(y)
    idx, _ = find_peaks(a, distance=max(int(0.3*P/dt), 1), prominence=0.01*a.max())
    if len(idx) < 4:
        return tu, a
    tq = np.linspace(tu[idx[0]], tu[idx[-1]], n_pts)
    return tq, PchipInterpolator(tu[idx], a[idx])(tq)

METH_F2 = ["Reference", "DDNN", "Fully scaled PINN"]      # drawing order (PINN on top)
STY     = {"Reference": ST_NUM, "DDNN": {**ST_DDNN}, "Fully scaled PINN": {**ST_PINN}}
ylabs   = {"h": r"$A_h$", "alpha": r"$A_\alpha$"}
f_ref   = {s: freq_fft(u_(U_ex_i[:, k])) for s, k in STATES}
P_ref   = {s: 1.0/f_ref[s] for s, _ in STATES}
f_max   = 2.5*max(f_ref.values())

fig2 = plt.figure(figsize=(6.27, 6.27))
root = fig2.add_gridspec(2, 1, left=0.12, right=0.97, bottom=0.1, top=0.94,
                         hspace=0.20)                      # gap between the two pairs
g_env  = root[0].subgridspec(2, 1, hspace=0.15)
g_spec = root[1].subgridspec(2, 1, hspace=0.15)

def tag(ax, s):
    ax.text(-0.1, 1.02, s, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold")

# ---- (a,b): amplitude envelopes, shared t-axis ----
ax_env = []
for r, (sname, k) in enumerate(STATES):
    ax = fig2.add_subplot(g_env[r], sharex=ax_env[0] if ax_env else None)
    for meth in METH_F2:
        tq, aq = envelope_smooth(u_(SERIES[meth][0][:, k]), P_ref[sname])
        ax.plot(tq, aq, **STY[meth])
    if LOG_ENVELOPE: ax.set_yscale("log")
    ax.set_xlim(tu[0], tu[-1])
    ax.set_ylabel(ylabs[sname]); style(ax, nticks=4)
    if LOG_ENVELOPE:                                  # semilog: proper log ticks
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(2, 0.4)))  # label minor ticks when range < 1 decade
    else:
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))   # "round" ticks
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    tag(ax, "(%s)" % "ab"[r])
    ax_env.append(ax)
plt.setp(ax_env[0].get_xticklabels(), visible=False)
ax_env[1].set_xlabel(r"$t\,(\mathrm{s})$")

# ---- (c,d): normalised spectra, shared f-axis ----
ax_spec = []
for r, (sname, k) in enumerate(STATES):
    ax = fig2.add_subplot(g_spec[r], sharex=ax_spec[0] if ax_spec else None)
    Smax = spec_store[(sname, "Reference")][1].max()
    for meth in METH_F2:
        fq, S = spec_store[(sname, meth)]
        ax.plot(fq, S/Smax, **STY[meth])
    ax.axvline(f_ref[sname], color="0.35", lw=0.7, ls=":", zorder=0)
    ax.text(f_ref[sname] + 0.02*f_max, 1.05, r"$f_{\mathrm{ref}}=%.2f\ \mathrm{Hz}$" % f_ref[sname],
            ha="left", va="top", fontsize=10)
    ax.set_xlim(0, f_max); ax.set_ylim(0, 1.1)
    sym = "h" if sname == "h" else r"\alpha"
    ax.set_ylabel(r"$|\hat{%s}|/|\hat{%s}|_{\max}^{\,ref}$" % (sym, sym)); style(ax, nticks=4)
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    tag(ax, "(%s)" % "cd"[r])
    ax_spec.append(ax)
plt.setp(ax_spec[0].get_xticklabels(), visible=False)
ax_spec[1].set_xlabel(r"$f\,(\mathrm{Hz})$")

fig2.align_ylabels(ax_env + ax_spec)
TICK_FS, LABEL_FS = 10, 14          # فونت اعداد تیک / فونت لیبل محورها
for ax in ax_env + ax_spec:
    ax.tick_params(which="major", labelsize=TICK_FS, width=1.3, length=6)
    ax.tick_params(which="minor", width=0.9, length=3.5)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FS)   # عبارت ×10⁻³ بالای محور
fig2.legend(legend_handles[:3], LABELS[:3], loc="upper center", bbox_to_anchor=(0.55, 0.995),
            ncol=3, frameon=False, handlelength=2.4, columnspacing=1.8, handletextpad=0.6)
plt.show()
fig2.savefig(f"figures/Figure_R15_envelope_spectrum_{CASE_TAG}.png", dpi=600, bbox_inches="tight")
fig2.savefig(f"figures/Figure_R15_envelope_spectrum_{CASE_TAG}.pdf", bbox_inches="tight")
print("\nSaved: tables/TableR15_*_%s.csv and figures/Figure_R15_envelope_spectrum_%s.(png|pdf)" % (CASE_TAG, CASE_TAG))
