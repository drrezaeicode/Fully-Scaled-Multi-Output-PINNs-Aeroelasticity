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
sys.path.append(os.path.abspath("../pitch-plunge-2d-u23-pinns-train-linspace"))


from Pitch_Plunge_2D_U_23_pinns_train import PINNs_MLPs 



from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library



loaded_workspace = load_globals("../pitch-plunge-2d-u23-pinns-train-linspace/pitch_plunge_2D_U_23_pinns_train.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u23-pinns-train-linspace/pitch_plunge_2D_U_23_pinns_network.pth", map_location=device, weights_only=True))
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
t_data_tensor_linspace = t_data_tensor
u_data_tensor_linspace = u_data_tensor


loaded_workspace = load_globals("../pitch-plunge-2d-u23-pinns-train-uniform-random/pitch_plunge_2D_U_23_pinns_train_random.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u23-pinns-train-uniform-random/pitch_plunge_2D_U_23_pinns_network_random.pth", map_location=device, weights_only=True))
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
t_data_tensor_random = t_data_tensor
u_data_tensor_random = u_data_tensor











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

plt.close("all")




# Example 0 : Figure 0.a  ==================================================================================================
fig_0 = plt.figure(figsize=(3.35, 2.7));
fig_0.subplots_adjust(hspace=0.0, wspace=0.0)
ax_0a = fig_0.add_subplot(211);fig_0.subplots_adjust(left=0.18, bottom=0.18)
ax_0b = fig_0.add_subplot(212);fig_0.subplots_adjust(left=0.18, bottom=0.18)
ax_0a.scatter(t_data_tensor_linspace.detach().cpu().numpy(),np.zeros_like(t_data_tensor_linspace.detach().cpu().numpy()),label=r"Linear Spaced",s=4,marker="o",color="#1E90FF",alpha=1,zorder=3)#
ax_0b.scatter(t_data_tensor_random.detach().cpu().numpy(),np.zeros_like(t_data_tensor_random.detach().cpu().numpy())    ,label=r"Uniform Random",s=4,marker="d",color="black",alpha=1,zorder=3)#
ax_0b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_0a.tick_params(direction='out', length=0.001, width=2, colors='w',
               grid_color='r', grid_alpha=0.5)
ax_0b.tick_params(axis="y",direction='out', length=0.001, width=2, colors='w',
               grid_color='r', grid_alpha=0.5)
fig_0.legend(loc="lower center",bbox_to_anchor=(0.52, .9),ncol=2,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_0b.set_xlim([0,2])
ax_0a.set_xlim([0,2])
plt.show()
fig_0.savefig("figures/Ex2_fig_0.png", dpi=600, bbox_inches='tight')
fig_0.savefig("figures/Ex2_fig_0.pdf", bbox_inches="tight")


#==========================================================================================================================
error_linspace     = u_exac_tensor - u_pred_tensor
error_random       = u_exac_tensor - u_pred_tensor_d
error_linspace_dot = u_exac_dot_tensor - u_pred_dot_tensor
error_random_dot   = u_exac_dot_tensor - u_pred_dot_tensor_d

# Example 0 : Figure 1.a  ==================================================================================================
fig_1a = plt.figure(figsize=(3.35, 2.7));ax_1a = fig_1a.add_subplot();fig_1a.subplots_adjust(left=0.18, bottom=0.18)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_linspace[:, 0].detach().cpu().numpy(),label=r"Linear Spaced",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_random[:, 0].detach().cpu().numpy(),label=r"Uniform Random",color="black",linewidth=1.0,linestyle="-.", zorder=2)
ax_1a.set_xlim([0,2])
ax_1a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1a.set_ylabel(r"$e_h$")
ax_1a.tick_params(direction="in", length=4, width=0.8)
ax_1a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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

fig_1a.savefig("figures/Ex2_fig_1a.png", dpi=600, bbox_inches='tight')
fig_1a.savefig("figures/Ex2_fig_1a.pdf", bbox_inches="tight")


# Example 0 : Figure 1.b  ==================================================================================================
fig_1b = plt.figure(figsize=(3.35, 2.7));ax_1b = fig_1b.add_subplot();fig_1b.subplots_adjust(left=0.18, bottom=0.18)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_linspace[:, 1].detach().cpu().numpy(),label=r"Linear Spaced",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_random[:, 1].detach().cpu().numpy(),label=r"Uniform Random",color="black",linewidth=1.0,linestyle="-.", zorder=2)
ax_1b.set_xlim([0,2])
ax_1b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1b.set_ylabel(r"$e_\alpha$")
ax_1b.tick_params(direction="in", length=4, width=0.8)
ax_1b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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

fig_1b.savefig("figures/Ex2_fig_1b.png", dpi=600, bbox_inches='tight')
fig_1b.savefig("figures/Ex2_fig_1b.pdf", bbox_inches="tight")



# Example 0 : Figure 1.c  ==================================================================================================
fig_1c = plt.figure(figsize=(3.35, 2.7));ax_1c = fig_1c.add_subplot();fig_1c.subplots_adjust(left=0.18, bottom=0.18)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_linspace_dot[:, 0].detach().cpu().numpy(),label=r"Linear Spaced",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_random_dot[:, 0].detach().cpu().numpy(),label=r"Uniform Random",color="black",linewidth=1.0,linestyle="-.", zorder=2)
ax_1c.set_xlim([0,2])
ax_1c.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1c.set_ylabel(r"$e_\dot{h}$")
ax_1c.tick_params(direction="in", length=4, width=0.8)
ax_1c.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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

fig_1c.savefig("figures/Ex2_fig_1c.png", dpi=600, bbox_inches='tight')
fig_1c.savefig("figures/Ex2_fig_1c.pdf", bbox_inches="tight")


# Example 0 : Figure 1.d  ==================================================================================================
fig_1d = plt.figure(figsize=(3.35, 2.7));ax_1d = fig_1d.add_subplot();fig_1d.subplots_adjust(left=0.18, bottom=0.18)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_linspace_dot[:, 1].detach().cpu().numpy(),label=r"Linear Spaced",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_random_dot[:, 1].detach().cpu().numpy(),label=r"Uniform Random",color="black",linewidth=1.0,linestyle="--", zorder=2)
ax_1d.set_xlim([0,2])
ax_1d.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1d.set_ylabel(r"$e_\dot{\alpha}$")
ax_1d.tick_params(direction="in", length=4, width=0.8)
ax_1d.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_1d.xaxis.set_major_locator(MaxNLocator(5))
ax_1d.yaxis.set_major_locator(MaxNLocator(5))
ax_1d.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1d.yaxis.set_minor_locator(AutoMinorLocator(2))
#ax_1d.set_ylim([-.06,.06])
ax_1d.tick_params(which="minor", direction="in", length=2, width=0.5)
#ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1d.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_1d.savefig("figures/Ex2_fig_1d.png", dpi=600, bbox_inches='tight')
fig_1d.savefig("figures/Ex2_fig_1d.pdf", bbox_inches="tight")






