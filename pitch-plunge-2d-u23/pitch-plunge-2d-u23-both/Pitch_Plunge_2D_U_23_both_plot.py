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
gc.collect()           # آزاد کردن حافظه اشغال شده توسط figure های قدیمی

import sys
import os
sys.path.append(os.path.abspath("../pitch-plunge-2d-u23-pinns-train"))
from Pitch_Plunge_2D_U_23_pinns_train import PINNs_MLPs  # اگر در ماژول جدا ذخیره شده
from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library



loaded_workspace = load_globals("../pitch-plunge-2d-u23-pinns-train/pitch_plunge_2D_U_23_pinns_train.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u23-pinns-train/pitch_plunge_2D_U_23_pinns_network.pth", map_location=device, weights_only=True))
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




loaded_workspace = load_globals("../pitch-plunge-2d-u23-data-train/pitch_plunge_2D_U_23_data_train.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u23-data-train/pitch_plunge_2D_U_23_data_network.pth", map_location=device, weights_only=True))
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












import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
# =======================
# Journal-accurate setup
# =======================
plt.rcParams.update({

    # Font
    "font.family": "Times New Roman",

    # MathText (Times New Roman)
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",

    # Font sizes (PRINT SIZE)
    "axes.labelsize": 10,     # Axis labels
    "xtick.labelsize": 8,     # Tick labels
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Colors
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

# Example 2 : Figure 1.a  =======================
plt.close("all")
fig_1a = plt.figure(figsize=(3.35, 2.7));ax_1a = fig_1a.add_subplot();fig_1a.subplots_adjust(left=0.18, bottom=0.18)

ax_1a.plot(t_exac_tensor.detach().cpu().numpy(),u_exac_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$",color="#1E90FF",linewidth=1.5,linestyle="-",zorder=0)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),u_pred_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black",linewidth=1.0,linestyle="--",zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),u_pred_tensor_d[:, 0].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C",linewidth=1.0,linestyle="-.", zorder=2)
ax_1a.scatter(t_data_tensor.detach().cpu().numpy(),u_data_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Observed}$",s=25, marker="o", facecolors="white", edgecolors="#E69F00",
           linewidths=1.2, zorder=5)
#ax_1a.scatter(t_ini_tensor.detach().cpu().numpy(),u_ini_tensor[:, 0].detach().cpu().numpy()                               ,s=4,marker="s", color="#8A2BE2",alpha=1,zorder=3)
ax_1a.set_xlim([-.01,2.01])
# s=25, marker="o", facecolors="white", edgecolors="#E69F00",
#            linewidths=1.2, zorder=5)


ax_1a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1a.set_ylabel(r"$h$")

ax_1a.tick_params(direction="in", length=4, width=0.8)
ax_1a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=2,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

ax_1a.xaxis.set_major_locator(MaxNLocator(5))
ax_1a.yaxis.set_major_locator(MaxNLocator(5))
ax_1a.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1a.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_1a.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1a.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()


# Example 2 : Figure 1.b  =======================
fig_1b = plt.figure(figsize=(3.35, 2.7));ax_1b = fig_1b.add_subplot();fig_1b.subplots_adjust(left=0.18, bottom=0.18)

ax_1b.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor_d[:, 1].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=2)
ax_1b.scatter(t_data_tensor.detach().cpu().numpy(),u_data_tensor[:, 1].detach().cpu().numpy(), label=r"$\mathrm{Observed}$"    ,s=2,marker="o",color="#FFD700",alpha=1,zorder=3)
#ax_1b.scatter(t_ini_tensor.detach().cpu().numpy(), u_ini_tensor[:, 1].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)
ax_1b.set_xlim([-.01,2.01])

ax_1b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1b.set_ylabel(r"$\alpha$")

ax_1b.tick_params(direction="in", length=4, width=0.8)
ax_1b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=2,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

ax_1b.xaxis.set_major_locator(MaxNLocator(5))
ax_1b.yaxis.set_major_locator(MaxNLocator(5))
ax_1b.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1b.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_1b.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1b.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

# Example 2 : Figure 1.c  =======================
fig_1c = plt.figure(figsize=(3.35, 2.7));ax_1c = fig_1c.add_subplot();fig_1c.subplots_adjust(left=0.18, bottom=0.18)

ax_1c.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_1c.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_1c.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_dot_tensor_d[:, 0].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=2)
#ax_1c.scatter(t_ini_tensor.detach().cpu().numpy(), du_ini_tensor[:, 0].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#FFD700", s=2,   marker='o',zorder=3)
ax_1c.set_xlim([-.01,2.01])

ax_1c.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1c.set_ylabel(r"$\dot{h}$")

ax_1c.tick_params(direction="in", length=4, width=0.8)
ax_1c.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

ax_1c.xaxis.set_major_locator(MaxNLocator(5))
ax_1c.yaxis.set_major_locator(MaxNLocator(5))
ax_1c.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1c.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_1c.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1c.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()


# Example 2 : Figure 1.d  =======================
fig_1d = plt.figure(figsize=(3.35, 2.7));ax_1d = fig_1d.add_subplot();fig_1d.subplots_adjust(left=0.18, bottom=0.18)

ax_1d.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_1d.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_1d.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_dot_tensor_d[:, 1].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=2)
#ax_1d.scatter(t_ini_tensor.detach().cpu().numpy(), du_ini_tensor[:, 1].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#FFD700", s=2,   marker='o',zorder=3)
ax_1d.set_xlim([-.01,2.01])

ax_1d.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1d.set_ylabel(r"$\dot{\alpha}$")

ax_1d.tick_params(direction="in", length=4, width=0.8)
ax_1d.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

ax_1d.xaxis.set_major_locator(MaxNLocator(5))
ax_1d.yaxis.set_major_locator(MaxNLocator(5))
ax_1d.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1d.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_1d.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1d.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

# Example 2 : Figure 2.a  =======================
fig_2a = plt.figure(figsize=(3.35, 3.35));ax_2a = fig_2a.add_subplot();fig_2a.subplots_adjust(left=0.18, bottom=0.18)

ax_2a.plot(  u_exac_tensor[:, 0].detach().cpu().numpy(),u_exac_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_2a.plot(  u_pred_tensor[:, 0].detach().cpu().numpy(),u_pred_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_2a.plot(  u_pred_tensor_d[:, 0].detach().cpu().numpy(),u_pred_dot_tensor_d[:, 0].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=2)

#ax_2a.scatter(u_ini_tensor[:,0].detach().cpu().numpy(),du_ini_tensor[:,0].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#FFD700", s=5,   marker='s',zorder=3)


ax_2a.set_xlabel(r"$h$")
ax_2a.set_ylabel(r"$\dot{h}$")

ax_2a.tick_params(direction="in", length=4, width=0.8)
ax_2a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

ax_2a.xaxis.set_major_locator(MaxNLocator(5))
ax_2a.yaxis.set_major_locator(MaxNLocator(5))
ax_2a.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_2a.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_2a.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_2a.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

# Example 2 : Figure 2.b  =======================
fig_2b = plt.figure(figsize=(3.35, 3.35));ax_2b = fig_2b.add_subplot();fig_2b.subplots_adjust(left=0.18, bottom=0.18)

ax_2b.plot(  u_exac_tensor[:, 1].detach().cpu().numpy(),u_exac_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Numerical}$"    ,color="#1E90FF",linewidth=1.5                ,zorder=0)
ax_2b.plot(  u_pred_tensor[:, 1].detach().cpu().numpy(),u_pred_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Fully\ Scaled\ PINN}$",color="black", linestyle="--",linewidth=1.0,zorder=1)
ax_2b.plot(  u_pred_tensor_d[:, 1].detach().cpu().numpy(),u_pred_dot_tensor_d[:, 1].detach().cpu().numpy(),label=r"$\mathrm{DDNN}$",color="#DC143C", linestyle="-.",linewidth=1.0,zorder=1)
#ax_2b.scatter(u_ini_tensor[:, 1].detach().cpu().numpy(),du_ini_tensor[:,1].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#FFD700", s=5,   marker='s',zorder=3)
a  =  u_exac_tensor[:, 1].detach().cpu().numpy()
ad = u_exac_dot_tensor[:, 1].detach().cpu().numpy()
# n_arrow = 200
# eps = 1  # طول خیلی کوتاه فلش (مماس)
# for i in range(0, len(a) - eps, n_arrow):
#     ax_2b.annotate(
#         "",
#         xy=(a[i + eps], ad[i + eps]),
#         xytext=(a[i], ad[i]),
#         arrowprops=dict(
#             arrowstyle="->",
#             color="#1f77b4",
#             lw=0.8,
#             mutation_scale=6,   # اندازه سر فلش
#             shrinkA=0,
#             shrinkB=0
#         ),
#         zorder=2
#     )
    
ax_2b.set_xlabel(r"$\alpha$")
ax_2b.set_ylabel(r"$\dot{\alpha}$")

ax_2b.tick_params(direction="in", length=4, width=0.8)
ax_2b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

ax_2b.xaxis.set_major_locator(MaxNLocator(5))
ax_2b.yaxis.set_major_locator(MaxNLocator(5))
ax_2b.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_2b.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_2b.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_2b.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()




fig_1a.savefig("figures/Figure_9a.png", dpi=600, bbox_inches='tight')
fig_1a.savefig("figures/Figure_9a.pdf", bbox_inches="tight")

fig_1b.savefig("figures/Figure_9b.png", dpi=600, bbox_inches='tight')
fig_1b.savefig("figures/Figure_9b.pdf", bbox_inches="tight")

fig_1c.savefig("figures/Figure_9c.png", dpi=600, bbox_inches='tight')
fig_1c.savefig("figures/Figure_9c.pdf", bbox_inches="tight")

fig_1d.savefig("figures/Figure_9d.png", dpi=600, bbox_inches='tight')
fig_1d.savefig("figures/Figure_9d.pdf", bbox_inches="tight")

fig_2a.savefig("figures/Figure_9e.png", dpi=600, bbox_inches='tight')
fig_2a.savefig("figures/Figure_9e.pdf", bbox_inches="tight")

fig_2b.savefig("figures/Figure_9f.png", dpi=600, bbox_inches='tight')
fig_2b.savefig("figures/Figure_9f.pdf", bbox_inches="tight")
