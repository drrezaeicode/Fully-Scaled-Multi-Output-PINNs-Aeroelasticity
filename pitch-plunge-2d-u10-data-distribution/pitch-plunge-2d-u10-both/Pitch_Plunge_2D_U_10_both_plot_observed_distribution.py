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
sys.path.append(os.path.abspath("../pitch-plunge-2d-u10-pinns-train"))


from Pitch_Plunge_2D_U_10_pinns_train_linspace import PINNs_MLPs 



from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_train_linspace.pth", scope=globals())
model_linspace = PINNs_MLPs([1,128,128,2]).to(device) 
model_linspace.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_network_linspace.pth", map_location=device, weights_only=True))
model_linspace.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_linspace = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_linspace = model_linspace(t_test_scaled_tensor_linspace)  # (N,2)
u_pred_dot_scaled_tensor_linspace = torch.zeros_like(u_pred_scaled_tensor_linspace)
u_pred_dot_scaled_tensor_linspace[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_linspace[:, 0:1], t_test_scaled_tensor_linspace, grad_outputs=torch.ones_like(u_pred_scaled_tensor_linspace[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_linspace[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_linspace[:, 1:2], t_test_scaled_tensor_linspace, grad_outputs=torch.ones_like(u_pred_scaled_tensor_linspace[:, 0:1]), create_graph=True)[0]
u_pred_tensor_linspace = scaler.inverse_transformX(u_pred_scaled_tensor_linspace)
u_pred_dot_tensor_linspace = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_linspace)
t_data_tensor_linspace = t_data_tensor
u_data_tensor_linspace = u_data_tensor


loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_train_random.pth", scope=globals())
model_random = PINNs_MLPs([1,128,128,2]).to(device) 
model_random.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_network_random.pth", map_location=device, weights_only=True))
model_random.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_random = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_random = model_random(t_test_scaled_tensor_random)  # (N,2)
u_pred_dot_scaled_tensor_random = torch.zeros_like(u_pred_scaled_tensor_random)
u_pred_dot_scaled_tensor_random[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_random[:, 0:1], t_test_scaled_tensor_random, grad_outputs=torch.ones_like(u_pred_scaled_tensor_random[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_random[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_random[:, 1:2], t_test_scaled_tensor_random, grad_outputs=torch.ones_like(u_pred_scaled_tensor_random[:, 0:1]), create_graph=True)[0]
u_pred_tensor_random = scaler.inverse_transformX(u_pred_scaled_tensor_random)
u_pred_dot_tensor_random = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_random)
t_data_tensor_random = t_data_tensor
u_data_tensor_random = u_data_tensor


loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_train_lhs.pth", scope=globals())
model_lhs = PINNs_MLPs([1,128,128,2]).to(device) 
model_lhs.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_network_lhs.pth", map_location=device, weights_only=True))
model_lhs.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_lhs = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_lhs = model_lhs(t_test_scaled_tensor_lhs)  # (N,2)
u_pred_dot_scaled_tensor_lhs = torch.zeros_like(u_pred_scaled_tensor_lhs)
u_pred_dot_scaled_tensor_lhs[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_lhs[:, 0:1], t_test_scaled_tensor_lhs, grad_outputs=torch.ones_like(u_pred_scaled_tensor_lhs[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_lhs[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_lhs[:, 1:2], t_test_scaled_tensor_lhs, grad_outputs=torch.ones_like(u_pred_scaled_tensor_lhs[:, 0:1]), create_graph=True)[0]
u_pred_tensor_lhs = scaler.inverse_transformX(u_pred_scaled_tensor_lhs)
u_pred_dot_tensor_lhs = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_lhs)
t_data_tensor_lhs = t_data_tensor
u_data_tensor_lhs = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_train_sobol.pth", scope=globals())
model_sobol = PINNs_MLPs([1,128,128,2]).to(device) 
model_sobol.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train/pitch_plunge_2D_U_10_pinns_network_sobol.pth", map_location=device, weights_only=True))
model_sobol.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_sobol = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_sobol = model_sobol(t_test_scaled_tensor_sobol)  # (N,2)
u_pred_dot_scaled_tensor_sobol = torch.zeros_like(u_pred_scaled_tensor_sobol)
u_pred_dot_scaled_tensor_sobol[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_sobol[:, 0:1], t_test_scaled_tensor_sobol, grad_outputs=torch.ones_like(u_pred_scaled_tensor_sobol[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_sobol[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_sobol[:, 1:2], t_test_scaled_tensor_sobol, grad_outputs=torch.ones_like(u_pred_scaled_tensor_sobol[:, 0:1]), create_graph=True)[0]
u_pred_tensor_sobol = scaler.inverse_transformX(u_pred_scaled_tensor_sobol)
u_pred_dot_tensor_sobol = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_sobol)
t_data_tensor_sobol = t_data_tensor
u_data_tensor_sobol = u_data_tensor


# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# from matplotlib.ticker import AutoMinorLocator
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


colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
]

# Example 0 : Figure 0.a  ==================================================================================================
fig0 = plt.figure(figsize=(3.35, 2.7));ax0 = fig0.add_subplot();fig0.subplots_adjust(left=0.18, bottom=0.18)
ax0.scatter(t_data_tensor_linspace.detach().cpu().numpy(),np.ones_like(t_data_tensor_linspace.detach().cpu().numpy())*1,label=r"UG",   s=4,marker="o",color=colors[0],alpha=1,zorder=0)#
ax0.scatter(t_data_tensor_random.detach().cpu().numpy(),np.ones_like(t_data_tensor_random.detach().cpu().numpy())*2    ,label=r"UR",   s=4,marker="s",color=colors[1],alpha=1,zorder=1)#
ax0.scatter(t_data_tensor_lhs.detach().cpu().numpy(),np.ones_like(t_data_tensor_lhs.detach().cpu().numpy())*3          ,label=r"LHS",  s=4,marker="^",color=colors[2],alpha=1,zorder=2)#
ax0.scatter(t_data_tensor_sobol.detach().cpu().numpy(),np.ones_like(t_data_tensor_sobol.detach().cpu().numpy())*4      ,label=r"Sobol",s=4,marker="d",color=colors[3],alpha=1,zorder=3)#


ax0.set_xlabel(r"$t\,(\mathrm{s})$")
ax0.tick_params(direction='out',axis='y', length=0.001, width=2, colors='w', grid_color='r', grid_alpha=0.5)
ax0.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax0.xaxis.set_major_locator(MaxNLocator(5))
#ax0.yaxis.set_major_locator(MaxNLocator(5))
ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
#ax0.yaxis.set_minor_locator(AutoMinorLocator(2))
ax0.tick_params(which="minor", direction="out", length=2, width=0.5)
ax0.grid(True,which="major",axis='x', linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax0.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

ax0.set_ylim([0,5])
ax0.set_xlim([0,2])



plt.show()
fig0.savefig("figures/Ex2_fig_0.png", dpi=600, bbox_inches='tight')
fig0.savefig("figures/Ex2_fig_0.pdf", bbox_inches="tight")


#==========================================================================================================================
error_linspace     = u_exac_tensor - u_pred_tensor_linspace
error_random       = u_exac_tensor - u_pred_tensor_random
error_lhs          = u_exac_tensor - u_pred_tensor_lhs
error_sobol        = u_exac_tensor - u_pred_tensor_sobol

error_linspace_dot = u_exac_dot_tensor - u_pred_dot_tensor_linspace
error_random_dot   = u_exac_dot_tensor - u_pred_dot_tensor_random
error_lhs_dot      = u_exac_dot_tensor - u_pred_dot_tensor_lhs
error_sobol_dot   = u_exac_dot_tensor - u_pred_dot_tensor_sobol

# Example 0 : Figure 1.a  ==================================================================================================
fig_1a = plt.figure(figsize=(3.35, 2.7));ax_1a = fig_1a.add_subplot();fig_1a.subplots_adjust(left=0.18, bottom=0.18)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_linspace[:, 0].detach().cpu().numpy(),label=r"UG",color="#009E73",linewidth=1.2,linestyle="-",zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_random[:, 0].detach().cpu().numpy(),label=r"UR",color="#0072B2",linewidth=1.2,linestyle="--", zorder=2)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_lhs[:, 0].detach().cpu().numpy(),label=r"LHS",color="#D55E00",linewidth=1.2,linestyle="-.", zorder=2)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_sobol[:, 0].detach().cpu().numpy(),label=r"Sobol",color="#4D4D4D",linewidth=1.2,linestyle=":", zorder=2)
ax_1a.set_xlim([0,2])
#ax_1a.set_ylim([-.00002,.00002])
ax_1a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1a.set_ylabel(r"$e_h$")
ax_1a.tick_params(direction="out", length=4, width=0.8)
ax_1a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_1a.xaxis.set_major_locator(MaxNLocator(5))
ax_1a.yaxis.set_major_locator(MaxNLocator(5))
ax_1a.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1a.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_1a.tick_params(which="minor", direction="out", length=2, width=0.5)
ax_1a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1a.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_1a.savefig("figures/Ex2_fig_1a.png", dpi=600, bbox_inches='tight')
fig_1a.savefig("figures/Ex2_fig_1a.pdf", bbox_inches="tight")


# Example 0 : Figure 1.b  ==================================================================================================
fig_1b = plt.figure(figsize=(3.35, 2.7));ax_1b = fig_1b.add_subplot();fig_1b.subplots_adjust(left=0.18, bottom=0.18)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_linspace[:, 1].detach().cpu().numpy(),label=r"UG",color="#009E73",linewidth=1.2,linestyle="-",zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_random[:, 1].detach().cpu().numpy(),label=r"UR",color="#0072B2",linewidth=1.2,linestyle="--", zorder=2)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_lhs[:, 1].detach().cpu().numpy(),label=r"LHS",color="#D55E00",linewidth=1.2,linestyle="-.", zorder=2)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_sobol[:, 1].detach().cpu().numpy(),label=r"Sobol",color="#4D4D4D",linewidth=1.2,linestyle=":", zorder=2)
ax_1b.set_xlim([0,2])
ax_1b.set_ylim([-.001,.001])
ax_1b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1b.set_ylabel(r"$e_\alpha$")
ax_1b.tick_params(direction="out", length=4, width=0.8)
ax_1b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_1b.xaxis.set_major_locator(MaxNLocator(5))
ax_1b.yaxis.set_major_locator(MaxNLocator(5))
ax_1b.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1b.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_1b.tick_params(which="minor", direction="out", length=2, width=0.5)
ax_1b.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1b.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_1b.savefig("figures/Ex2_fig_1b.png", dpi=600, bbox_inches='tight')
fig_1b.savefig("figures/Ex2_fig_1b.pdf", bbox_inches="tight")



# Example 0 : Figure 1.c  ==================================================================================================
fig_1c = plt.figure(figsize=(3.35, 2.7));ax_1c = fig_1c.add_subplot();fig_1c.subplots_adjust(left=0.18, bottom=0.18)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_linspace_dot[:, 0].detach().cpu().numpy(),label=r"UG",color="#009E73",linewidth=1.2,linestyle="-",zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_random_dot[:, 0].detach().cpu().numpy(),label=r"UR",color="#0072B2",linewidth=1.2,linestyle="--", zorder=2)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_lhs_dot[:, 0].detach().cpu().numpy(),label=r"LHS",color="#D55E00",linewidth=1.2,linestyle="-.", zorder=3)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_sobol_dot[:, 0].detach().cpu().numpy(),label=r"Sobol",color="#4D4D4D",linewidth=1.2,linestyle=":", zorder=4)
ax_1c.set_xlim([0,2])
ax_1c.set_ylim([-.002,.002])
ax_1c.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1c.set_ylabel(r"$e_\dot{h}$")
ax_1c.tick_params(direction="out", length=4, width=0.8)
ax_1c.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_1c.xaxis.set_major_locator(MaxNLocator(5))
ax_1c.yaxis.set_major_locator(MaxNLocator(5))
ax_1c.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1c.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_1c.tick_params(which="minor", direction="out", length=2, width=0.5)
ax_1c.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1c.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_1c.savefig("figures/Ex2_fig_1c.png", dpi=600, bbox_inches='tight')
fig_1c.savefig("figures/Ex2_fig_1c.pdf", bbox_inches="tight")


# Example 0 : Figure 1.d  ==================================================================================================
fig_1d = plt.figure(figsize=(3.35, 2.7));ax_1d = fig_1d.add_subplot();fig_1d.subplots_adjust(left=0.18, bottom=0.18)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_linspace_dot[:, 1].detach().cpu().numpy(),label=r"UG",color="#009E73",linewidth=1.2,linestyle="-",zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_random_dot[:, 1].detach().cpu().numpy(),label=r"UR",color="#0072B2",linewidth=1.2,linestyle="--", zorder=2)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_lhs_dot[:, 1].detach().cpu().numpy(),label=r"LHS",color="#D55E00",linewidth=1.2,linestyle="-.", zorder=3)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_sobol_dot[:, 1].detach().cpu().numpy(),label=r"Sobol",color="#4D4D4D",linewidth=1.2,linestyle=":", zorder=4)
ax_1d.set_xlim([0,2])
ax_1d.set_ylim([-.05,.05])
ax_1d.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1d.set_ylabel(r"$e_\dot{\alpha}$")
ax_1d.tick_params(direction="out", length=4, width=0.8)
ax_1d.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=5,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_1d.xaxis.set_major_locator(MaxNLocator(5))
ax_1d.yaxis.set_major_locator(MaxNLocator(5))
ax_1d.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1d.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_1d.set_ylim([-.06,.06])
ax_1d.tick_params(which="minor", direction="out", length=2, width=0.5)
ax_1d.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1d.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_1d.savefig("figures/Ex2_fig_1d.png", dpi=600, bbox_inches='tight')
fig_1d.savefig("figures/Ex2_fig_1d.pdf", bbox_inches="tight")







#=========================================================================

distribution = ["UG", "UR", "LH", "SO"]

u_pred_dict = {
    "uniform_grid": u_pred_tensor_linspace,
    "uniform_random": u_pred_tensor_random,
    "latin_hypercube": u_pred_tensor_lhs,
    "sobol": u_pred_tensor_sobol,
}

u_pred_dot_dict = {
    "uniform_grid": u_pred_dot_tensor_linspace,
    "uniform_random": u_pred_dot_tensor_random,
    "latin_hypercube": u_pred_dot_tensor_lhs,
    "sobol": u_pred_dot_tensor_sobol,
}

mse_list, mae_list, rmse_list = [], [], []
mse_dot_list, mae_dot_list, rmse_dot_list = [], [], []

for key in u_pred_dict.keys():
    print(key)
    pred = u_pred_dict[key]
    pred_dot = u_pred_dot_dict[key]

    mse = ((u_exac_tensor - pred) ** 2).mean(dim=0)
    mae = torch.abs(u_exac_tensor - pred).mean(dim=0)
    rmse = torch.sqrt(mse)

    mse_dot = ((u_exac_dot_tensor - pred_dot) ** 2).mean(dim=0)
    mae_dot = torch.abs(u_exac_dot_tensor - pred_dot).mean(dim=0)
    rmse_dot = torch.sqrt(mse_dot)

    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)

    mse_dot_list.append(mse_dot)
    mae_dot_list.append(mae_dot)
    rmse_dot_list.append(rmse_dot)

mse_arr      = torch.stack(mse_list).detach().cpu().numpy()
mae_arr      = torch.stack(mae_list).detach().cpu().numpy()
rmse_arr     = torch.stack(rmse_list).detach().cpu().numpy()

mse_dot_arr  = torch.stack(mse_dot_list).detach().cpu().numpy()
mae_dot_arr  = torch.stack(mae_dot_list).detach().cpu().numpy()
rmse_dot_arr = torch.stack(rmse_dot_list).detach().cpu().numpy()

# Components
mse_h, mse_alpha = mse_arr[:, 0], mse_arr[:, 1]
mae_h, mae_alpha = mae_arr[:, 0], mae_arr[:, 1]
rmse_h, rmse_alpha = rmse_arr[:, 0], rmse_arr[:, 1]

mse_h_dot, mse_alpha_dot = mse_dot_arr[:, 0], mse_dot_arr[:, 1]
mae_h_dot, mae_alpha_dot = mae_dot_arr[:, 0], mae_dot_arr[:, 1]
rmse_h_dot, rmse_alpha_dot = rmse_dot_arr[:, 0], rmse_dot_arr[:, 1]


# Relative L2
rel_l2_list, rel_l2_dot_list = [], []

for key in u_pred_dict.keys():

    pred = u_pred_dict[key]
    pred_dot = u_pred_dot_dict[key]

    rel_l2 = (
        torch.linalg.norm(u_exac_tensor - pred, dim=0)
        / torch.linalg.norm(u_exac_tensor, dim=0)
    )

    rel_l2_dot = (
        torch.linalg.norm(u_exac_dot_tensor - pred_dot, dim=0)
        / torch.linalg.norm(u_exac_dot_tensor, dim=0)
    )

    rel_l2_list.append(rel_l2)
    rel_l2_dot_list.append(rel_l2_dot)

rel_l2_arr = torch.stack(rel_l2_list).detach().cpu().numpy()
rel_l2_dot_arr = torch.stack(rel_l2_dot_list).detach().cpu().numpy()

rel_l2_h, rel_l2_alpha = rel_l2_arr[:, 0], rel_l2_arr[:, 1]
rel_l2_h_dot, rel_l2_alpha_dot = rel_l2_dot_arr[:, 0], rel_l2_dot_arr[:, 1]






# Labels3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
methods = ['UG', 'UR', 'LHS', 'Sobol']

# Relative L2 errors
h_vals         = rel_l2_h
alpha_vals     = rel_l2_alpha
h_dot_vals     = rel_l2_h_dot
alpha_dot_vals = rel_l2_alpha_dot

# Group labels (LaTeX)
groups = [
    r'$h$',
    r'$\alpha$',
    r'$\dot{h}$',
    r'$\dot{\alpha}$'
]

# Combine data
data = np.array([
    h_vals,
    alpha_vals,
    h_dot_vals,
    alpha_dot_vals
])

# Figure

fig_2a = plt.figure(figsize=(3.35, 2.7));ax_2a = fig_2a.add_subplot();fig_2a.subplots_adjust(left=0.18, bottom=0.18)

# ax_2a = fig_2a.add_subplot()

# fig_2a.subplots_adjust(
#     left=0.18,
#     bottom=0.18,
#     top=0.78
# )

# Bar positions
x = np.arange(len(groups))
width = 0.18

# Bars
ax_2a.bar(x - 1.5*width, data[:,0], width, label='UG')
ax_2a.bar(x - 0.5*width, data[:,1], width, label='UR')
ax_2a.bar(x + 0.5*width, data[:,2], width, label='LHS')
ax_2a.bar(x + 1.5*width, data[:,3], width, label='Sobol')

# Axes
ax_2a.set_ylabel(r'Relative $L_2$ Error')
ax_2a.set_ylim([0,.006])
ax_2a.set_xticks(x)
ax_2a.set_xticklabels(groups)

# Grid
#ax_2a.grid(True,linestyle='--', alpha=0.5)
ax_2a.grid(True, axis='y', which="major",linestyle=":",linewidth=0.4,alpha=0.6)

# Legend (outside, top, single row)
ax_2a.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    fontsize=8,
    frameon=False
)

plt.show()



fig_2a.savefig("figures/Ex2_fig_2a.png", dpi=600, bbox_inches='tight')
fig_2a.savefig("figures/Ex2_fig_2a.pdf", bbox_inches="tight")