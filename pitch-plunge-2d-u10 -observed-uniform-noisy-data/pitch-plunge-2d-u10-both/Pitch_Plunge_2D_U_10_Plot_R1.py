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
sys.path.append(os.path.abspath("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data"))


from Pitch_Plunge_2D_U_10_pinns_noisy_std_1_100 import PINNs_MLPs 



from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_0_100.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_0_100_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_0_100 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_0_100 = model(t_test_scaled_tensor_0_100)  # (N,2)
u_pred_dot_scaled_tensor_0_100 = torch.zeros_like(u_pred_scaled_tensor_0_100)
u_pred_dot_scaled_tensor_0_100[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_0_100[:, 0:1], t_test_scaled_tensor_0_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_0_100[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_0_100[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_0_100[:, 1:2], t_test_scaled_tensor_0_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_0_100[:, 0:1]), create_graph=True)[0]
u_pred_tensor_0_100 = scaler.inverse_transformX(u_pred_scaled_tensor_0_100)
u_pred_dot_tensor_0_100 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_0_100)
t_data_tensor_0_100 = t_data_tensor
u_data_tensor_0_100 = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_1_100.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_1_100_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_1_100 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_1_100 = model(t_test_scaled_tensor_1_100)  # (N,2)
u_pred_dot_scaled_tensor_1_100 = torch.zeros_like(u_pred_scaled_tensor_1_100)
u_pred_dot_scaled_tensor_1_100[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_1_100[:, 0:1], t_test_scaled_tensor_1_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_1_100[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_1_100[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_1_100[:, 1:2], t_test_scaled_tensor_1_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_1_100[:, 0:1]), create_graph=True)[0]
u_pred_tensor_1_100 = scaler.inverse_transformX(u_pred_scaled_tensor_1_100)
u_pred_dot_tensor_1_100 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_1_100)
t_data_tensor_1_100 = t_data_tensor
u_data_tensor_1_100 = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_5_100.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_5_100_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_5_100 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_5_100 = model(t_test_scaled_tensor_5_100)  # (N,2)
u_pred_dot_scaled_tensor_5_100 = torch.zeros_like(u_pred_scaled_tensor_5_100)
u_pred_dot_scaled_tensor_5_100[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_5_100[:, 0:1], t_test_scaled_tensor_5_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_5_100[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_5_100[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_5_100[:, 1:2], t_test_scaled_tensor_5_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_5_100[:, 0:1]), create_graph=True)[0]
u_pred_tensor_5_100 = scaler.inverse_transformX(u_pred_scaled_tensor_5_100)
u_pred_dot_tensor_5_100 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_5_100)
t_data_tensor_5_100 = t_data_tensor
u_data_tensor_5_100 = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_1_10.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_1_10_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_1_10 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_1_10 = model(t_test_scaled_tensor_1_10)  # (N,2)
u_pred_dot_scaled_tensor_1_10 = torch.zeros_like(u_pred_scaled_tensor_1_10)
u_pred_dot_scaled_tensor_1_10[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_1_10[:, 0:1], t_test_scaled_tensor_1_10, grad_outputs=torch.ones_like(u_pred_scaled_tensor_1_10[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_1_10[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_1_10[:, 1:2], t_test_scaled_tensor_1_10, grad_outputs=torch.ones_like(u_pred_scaled_tensor_1_10[:, 0:1]), create_graph=True)[0]
u_pred_tensor_1_10 = scaler.inverse_transformX(u_pred_scaled_tensor_1_10)
u_pred_dot_tensor_1_10 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_1_10)
t_data_tensor_1_10 = t_data_tensor
u_data_tensor_1_10 = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_15_100.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_15_100_network.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_15_100 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_15_100 = model(t_test_scaled_tensor_15_100)  # (N,2)
u_pred_dot_scaled_tensor_15_100 = torch.zeros_like(u_pred_scaled_tensor_15_100)
u_pred_dot_scaled_tensor_15_100[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_15_100[:, 0:1], t_test_scaled_tensor_15_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_15_100[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_15_100[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_15_100[:, 1:2], t_test_scaled_tensor_15_100, grad_outputs=torch.ones_like(u_pred_scaled_tensor_15_100[:, 0:1]), create_graph=True)[0]
u_pred_tensor_15_100 = scaler.inverse_transformX(u_pred_scaled_tensor_15_100)
u_pred_dot_tensor_15_100 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_15_100)
t_data_tensor_15_100 = t_data_tensor
u_data_tensor_15_100 = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_2_10.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-uniform-noisy-data/pitch_plunge_2D_U_10_pinns_noisy_std_2_10_network.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_2_10 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_2_10 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_2_10 = torch.zeros_like(u_pred_scaled_tensor_2_10)
u_pred_dot_scaled_tensor_2_10[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_2_10[:, 0:1], t_test_scaled_tensor_2_10, grad_outputs=torch.ones_like(u_pred_scaled_tensor_2_10[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_2_10[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_2_10[:, 1:2], t_test_scaled_tensor_2_10, grad_outputs=torch.ones_like(u_pred_scaled_tensor_2_10[:, 0:1]), create_graph=True)[0]
u_pred_tensor_2_10 = scaler.inverse_transformX(u_pred_scaled_tensor_2_10)
u_pred_dot_tensor_2_10 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_2_10)
t_data_tensor_2_10 = t_data_tensor
u_data_tensor_2_10 = u_data_tensor





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



'''
# Example 0 : Figure 0.a  ==================================================================================================
fig_0 = plt.figure(figsize=(3.35, 2.7));
fig_0.subplots_adjust(hspace=0.0, wspace=0.0)
ax_0a = fig_0.add_subplot(211);fig_0.subplots_adjust(left=0.18, bottom=0.18)
ax_0b = fig_0.add_subplot(212);fig_0.subplots_adjust(left=0.18, bottom=0.18)
ax_0a.scatter(t_data_tensor_30.detach().cpu().numpy(),np.zeros_like(t_data_tensor_30.detach().cpu().numpy()),label=r"Linear Spaced",s=4,marker="o",color="#1E90FF",alpha=1,zorder=3)#
ax_0b.scatter(t_data_tensor_30.detach().cpu().numpy(),np.zeros_like(t_data_tensor_30.detach().cpu().numpy()),label=r"Uniform Random",s=4,marker="d",color="black",alpha=1,zorder=3)#
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
'''

#==========================================================================================================================
error_0_100   = u_exac_tensor - u_pred_tensor_0_100
error_1_100   = u_exac_tensor - u_pred_tensor_1_100
error_5_100   = u_exac_tensor - u_pred_tensor_5_100
error_1_10    = u_exac_tensor - u_pred_tensor_1_10
error_15_100  = u_exac_tensor - u_pred_tensor_15_100
error_2_10   = u_exac_tensor - u_pred_tensor_2_10

error_dot_0_100   = u_exac_dot_tensor - u_pred_dot_tensor_0_100
error_dot_1_100   = u_exac_dot_tensor - u_pred_dot_tensor_1_100
error_dot_5_100   = u_exac_dot_tensor - u_pred_dot_tensor_5_100
error_dot_1_10    = u_exac_dot_tensor - u_pred_dot_tensor_1_10
error_dot_15_100  = u_exac_dot_tensor - u_pred_dot_tensor_15_100
error_dot_2_10     = u_exac_dot_tensor - u_pred_dot_tensor_2_10

"#0072B2"
linestyles = [(0, (5, 1, 1, 1,1,1)), "-", "--","-.",":",]
colors = ["#4D4D4D","#0072B2", "#D55E00","#009E73","#CC79A7"  , ]

colors = ["#4D4D4D","#0072B2","#D55E00","#009E73","#CC79A7"]

# Example 0 : Figure 1.a  ==================================================================================================
fig_1a = plt.figure(figsize=(3.35, 2.7));ax_1a = fig_1a.add_subplot();fig_1a.subplots_adjust(left=0.18, bottom=0.18)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(), error_0_100[:, 0].detach().cpu().numpy() ,label=r"$\sigma = 0\%$" ,color=colors[0],linewidth=1.0,linestyle=linestyles[0],zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(), error_1_10[:, 0].detach().cpu().numpy() ,label=r"$\sigma = 10\%$" ,color=colors[1],linewidth=1.0,linestyle=linestyles[1],zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(), error_2_10[:, 0].detach().cpu().numpy()  ,label=r"$\sigma = 20\%$",color=colors[2],linewidth=1.0,linestyle=linestyles[2],zorder=1)
#ax_1a.plot(t_test_tensor.detach().cpu().numpy(), error_2_10[:, 0].detach().cpu().numpy(),label=r"$\sigma = 20\%$",color=colors[3],linewidth=1.0,linestyle=linestyles[3],zorder=1)
#ax_1a.plot(t_test_tensor.detach().cpu().numpy(), error_2_10[:, 0].detach().cpu().numpy()  ,label=r"$\sigma = 20\%$",color=colors[3],linewidth=1.0,linestyle=linestyles[3],zorder=1)

ax_1a.set_xlim([0,2])
ax_1a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1a.set_ylabel(r"$e_\alpha$")
ax_1a.tick_params(direction="out", length=4, width=0.8)
ax_1a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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
ax_1b.plot(t_test_tensor.detach().cpu().numpy(), error_0_100[:, 1].detach().cpu().numpy() ,label=r"$\sigma = 0\%$" ,color=colors[0],linewidth=1.0,linestyle=linestyles[0],zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(), error_1_10[:, 1].detach().cpu().numpy() ,label=r"$\sigma = 10\%$" ,color=colors[1],linewidth=1.0,linestyle=linestyles[1],zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(), error_2_10[:, 1].detach().cpu().numpy()  ,label=r"$\sigma = 20\%$",color=colors[2],linewidth=1.0,linestyle=linestyles[2],zorder=1)
# ax_1b.plot(t_test_tensor.detach().cpu().numpy(), error_15_100[:, 1].detach().cpu().numpy(),label=r"$\sigma = 15\%$",color=colors[3],linewidth=1.0,linestyle=linestyles[3],zorder=1)
# ax_1b.plot(t_test_tensor.detach().cpu().numpy(), error_2_10[:, 1].detach().cpu().numpy()  ,label=r"$\sigma = 20\%$",color=colors[4],linewidth=1.0,linestyle=linestyles[4],zorder=1)

ax_1b.set_xlim([0,2])
ax_1b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1b.set_ylabel(r"$e_\alpha$")
ax_1b.tick_params(direction="out", length=4, width=0.8)
ax_1b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=6,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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
ax_1c.plot(t_test_tensor.detach().cpu().numpy(), error_dot_0_100[:, 0].detach().cpu().numpy() ,label=r"$\sigma = 0\%$" ,color=colors[0],linewidth=1.0,linestyle=linestyles[0],zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(), error_dot_1_10[:, 0].detach().cpu().numpy() ,label=r"$\sigma = 10\%$" ,color=colors[1],linewidth=1.0,linestyle=linestyles[1],zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(), error_dot_2_10[:, 0].detach().cpu().numpy()  ,label=r"$\sigma = 20\%$",color=colors[2],linewidth=1.0,linestyle=linestyles[2],zorder=1)

ax_1c.set_xlim([0,2])
ax_1c.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1c.set_ylabel(r"$e_\dot{h}$")
ax_1c.tick_params(direction="out", length=4, width=0.8)
ax_1c.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=6,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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
ax_1d.plot(t_test_tensor.detach().cpu().numpy(), error_dot_0_100[:, 1].detach().cpu().numpy() ,label=r"$\sigma = 0\%$" ,color=colors[0],linewidth=1.0,linestyle=linestyles[0],zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(), error_dot_1_10[:, 1].detach().cpu().numpy() ,label=r"$\sigma = 10\%$" ,color=colors[1],linewidth=1.0,linestyle=linestyles[1],zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(), error_dot_2_10[:, 1].detach().cpu().numpy()  ,label=r"$\sigma = 20\%$",color=colors[2],linewidth=1.0,linestyle=linestyles[2],zorder=1)

ax_1d.set_xlim([0,2])
ax_1d.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1d.set_ylabel(r"$e_\dot{\alpha}$")
ax_1d.tick_params(direction="out", length=4, width=0.8)
ax_1d.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_1d.xaxis.set_major_locator(MaxNLocator(5))
ax_1d.yaxis.set_major_locator(MaxNLocator(5))
ax_1d.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_1d.yaxis.set_minor_locator(AutoMinorLocator(2))
#ax_1d.set_ylim([-.06,.06])
ax_1d.tick_params(which="minor", direction="out", length=2, width=0.5)
ax_1d.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_1d.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_1d.savefig("figures/Ex2_fig_1d.png", dpi=600, bbox_inches='tight')
fig_1d.savefig("figures/Ex2_fig_1d.pdf", bbox_inches="tight")

# MSE vs observed data vs MSE ===========================================================================================

mse_0_100 = ((u_exac_tensor - u_pred_tensor_0_100)**2).mean(dim=0)#.unsqueeze(0)
mse_1_100 = ((u_exac_tensor - u_pred_tensor_1_100)**2).mean(dim=0)#.unsqueeze(0)
mse_5_100 = ((u_exac_tensor - u_pred_tensor_5_100)**2).mean(dim=0)#.unsqueeze(0)
mse_1_10  = ((u_exac_tensor - u_pred_tensor_1_10)**2).mean(dim=0)#.unsqueeze(0)
mse_15_100 = ((u_exac_tensor - u_pred_tensor_15_100)**2).mean(dim=0)#.unsqueeze(0)
mse_2_10   = ((u_exac_tensor - u_pred_tensor_2_10)**2).mean(dim=0)#.unsqueeze(0)

mse_dot_0_100   = ((u_exac_dot_tensor - u_pred_dot_tensor_0_100)**2).mean(dim=0)#.unsqueeze(0)
mse_dot_1_100   = ((u_exac_dot_tensor - u_pred_dot_tensor_1_100)**2).mean(dim=0)#.unsqueeze(0)
mse_dot_5_100   = ((u_exac_dot_tensor - u_pred_dot_tensor_5_100)**2).mean(dim=0)#.unsqueeze(0)
mse_dot_15_100  = ((u_exac_dot_tensor - u_pred_dot_tensor_15_100)**2).mean(dim=0)#.unsqueeze(0)
mse_dot_1_10    = ((u_exac_dot_tensor -  u_pred_dot_tensor_1_10)**2).mean(dim=0)#.unsqueeze(0)
mse_dot_2_10     = ((u_exac_dot_tensor -   u_pred_dot_tensor_2_10)**2).mean(dim=0)#.unsqueeze(0)


mse_h         = torch.stack([mse_0_100,mse_1_100,mse_5_100,mse_1_10,mse_15_100, mse_2_10], dim=0)[:,0].detach().cpu().numpy()
mse_alpha     = torch.stack([mse_0_100,mse_1_100,mse_5_100,mse_1_10,mse_15_100, mse_2_10], dim=0)[:,1].detach().cpu().numpy()
mse_h_dot     = torch.stack([mse_dot_0_100,mse_dot_1_100,mse_dot_5_100, mse_dot_1_10,mse_dot_15_100, mse_dot_2_10], dim=0)[:,0].detach().cpu().numpy()
mse_alpha_dot = torch.stack([mse_dot_0_100,mse_dot_1_100,mse_dot_5_100, mse_dot_1_10,mse_dot_15_100, mse_dot_2_10], dim=0)[:,1].detach().cpu().numpy()




# MAE vs observed data vs MAE ===========================================================================================
mae_0_100  = torch.abs(u_exac_tensor - u_pred_tensor_0_100).mean(dim=0)
mae_1_100  = torch.abs(u_exac_tensor - u_pred_tensor_1_100).mean(dim=0)
mae_5_100  = torch.abs(u_exac_tensor - u_pred_tensor_5_100).mean(dim=0)
mae_1_10   = torch.abs(u_exac_tensor - u_pred_tensor_1_10).mean(dim=0)
mae_15_100 = torch.abs(u_exac_tensor - u_pred_tensor_15_100).mean(dim=0)
mae_2_10    = torch.abs(u_exac_tensor - u_pred_tensor_2_10).mean(dim=0)

mae_dot_0_100  = torch.abs(u_exac_dot_tensor - u_pred_dot_tensor_0_100).mean(dim=0)
mae_dot_1_100  = torch.abs(u_exac_dot_tensor - u_pred_dot_tensor_1_100).mean(dim=0)
mae_dot_5_100  = torch.abs(u_exac_dot_tensor - u_pred_dot_tensor_5_100).mean(dim=0)
mae_dot_1_10   = torch.abs(u_exac_dot_tensor - u_pred_dot_tensor_1_10).mean(dim=0)
mae_dot_15_100 = torch.abs(u_exac_dot_tensor - u_pred_dot_tensor_15_100).mean(dim=0)
mae_dot_2_10    = torch.abs(u_exac_dot_tensor - u_pred_dot_tensor_2_10).mean(dim=0)


mae_h         = torch.stack([mae_0_100,mae_1_100,mae_5_100,mae_1_10,mae_15_100, mae_2_10], dim=0)[:,0].detach().cpu().numpy()
mae_alpha     = torch.stack([mae_0_100,mae_1_100,mae_5_100,mae_1_10,mae_15_100, mae_2_10], dim=0)[:,1].detach().cpu().numpy()
mae_h_dot     = torch.stack([mae_dot_0_100,mae_dot_1_100,mae_dot_5_100, mae_dot_1_10,mae_dot_15_100, mae_dot_2_10], dim=0)[:,0].detach().cpu().numpy()
mae_alpha_dot = torch.stack([mae_dot_0_100,mae_dot_1_100,mae_dot_5_100, mae_dot_1_10,mae_dot_15_100, mae_dot_2_10], dim=0)[:,1].detach().cpu().numpy()


# RSME vs observed data vs RSME ===========================================================================================

# RMSE (Root Mean Square Error)

rmse_0_100 = torch.sqrt(mse_0_100)
rmse_1_100 = torch.sqrt(mse_1_100)
rmse_5_100 = torch.sqrt(mse_5_100)
rmse_1_10  = torch.sqrt(mse_1_10)
rmse_15_100 = torch.sqrt(mse_15_100)
rmse_2_10   = torch.sqrt(mse_2_10)

rmse_dot_0_100 = torch.sqrt(mse_dot_0_100)
rmse_dot_1_100 = torch.sqrt(mse_dot_1_100)
rmse_dot_5_100 = torch.sqrt(mse_dot_5_100)
rmse_dot_1_10  = torch.sqrt(mse_dot_1_10)
rmse_dot_15_100 = torch.sqrt(mse_dot_15_100)
rmse_dot_2_10   = torch.sqrt(mse_dot_2_10)


rmse_h         = torch.stack([rmse_0_100,rmse_1_100,rmse_5_100,rmse_1_10,rmse_15_100, rmse_2_10], dim=0)[:,0].detach().cpu().numpy()
rmse_alpha     = torch.stack([rmse_0_100,rmse_1_100,rmse_5_100,rmse_1_10,rmse_15_100, rmse_2_10], dim=0)[:,1].detach().cpu().numpy()
rmse_h_dot     = torch.stack([rmse_dot_0_100,rmse_dot_1_100,rmse_dot_5_100, rmse_dot_1_10,rmse_dot_15_100, rmse_dot_2_10], dim=0)[:,0].detach().cpu().numpy()
rmse_alpha_dot = torch.stack([rmse_dot_0_100,rmse_dot_1_100,rmse_dot_5_100, rmse_dot_1_10,rmse_dot_15_100, rmse_dot_2_10], dim=0)[:,1].detach().cpu().numpy()


# Relative L2 Error =======================================================================================================

rel_l2_0_100  = torch.linalg.norm(u_exac_tensor - u_pred_tensor_0_100 , dim=0) / torch.linalg.norm( u_exac_tensor, dim=0)
rel_l2_1_100  = torch.linalg.norm(u_exac_tensor - u_pred_tensor_1_100 , dim=0) / torch.linalg.norm( u_exac_tensor, dim=0)
rel_l2_5_100  = torch.linalg.norm(u_exac_tensor - u_pred_tensor_5_100 , dim=0) / torch.linalg.norm( u_exac_tensor, dim=0)
rel_l2_1_10   = torch.linalg.norm(u_exac_tensor - u_pred_tensor_1_10  , dim=0) / torch.linalg.norm( u_exac_tensor, dim=0)
rel_l2_15_100 = torch.linalg.norm(u_exac_tensor - u_pred_tensor_15_100, dim=0) / torch.linalg.norm( u_exac_tensor, dim=0)
rel_l2_2_10   = torch.linalg.norm(u_exac_tensor - u_pred_tensor_2_10  , dim=0) / torch.linalg.norm( u_exac_tensor, dim=0)

rel_l2_dot_0_100  = torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_tensor_0_100 , dim=0) / torch.linalg.norm( u_exac_dot_tensor, dim=0)
rel_l2_dot_1_100  = torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_tensor_1_100 , dim=0) / torch.linalg.norm( u_exac_dot_tensor, dim=0)
rel_l2_dot_5_100  = torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_tensor_5_100 , dim=0) / torch.linalg.norm( u_exac_dot_tensor, dim=0)
rel_l2_dot_1_10   = torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_tensor_1_10  , dim=0) / torch.linalg.norm( u_exac_dot_tensor, dim=0)
rel_l2_dot_15_100 = torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_tensor_15_100, dim=0) / torch.linalg.norm( u_exac_dot_tensor, dim=0)
rel_l2_dot_2_10   = torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_tensor_2_10  , dim=0) / torch.linalg.norm( u_exac_dot_tensor, dim=0)


rel_l2_h         = torch.stack([rel_l2_0_100, rel_l2_1_100, rel_l2_5_100, rel_l2_1_10, rel_l2_15_100, rel_l2_2_10], dim=0)[:,0].detach().cpu().numpy()
rel_l2_alpha     = torch.stack([rel_l2_0_100, rel_l2_1_100, rel_l2_5_100, rel_l2_1_10, rel_l2_15_100, rel_l2_2_10], dim=0)[:,1].detach().cpu().numpy()
rel_l2_h_dot     = torch.stack([rel_l2_dot_0_100, rel_l2_dot_1_100, rel_l2_dot_5_100, rel_l2_dot_1_10, rel_l2_dot_15_100, rel_l2_dot_2_10], dim=0)[:,0].detach().cpu().numpy()
rel_l2_alpha_dot = torch.stack([rel_l2_dot_0_100, rel_l2_dot_1_100, rel_l2_dot_5_100, rel_l2_dot_1_10, rel_l2_dot_15_100, rel_l2_dot_2_10], dim=0)[:,1].detach().cpu().numpy()








noise = np.array([0,1,5,10,15,20])

# Example 0 : Figure 2.a  ==================================================================================================
fig_2a = plt.figure(figsize=(3.35, 2.7));ax_2a = fig_2a.add_subplot();fig_2a.subplots_adjust(left=0.18, bottom=0.18)
ax_2a.plot(noise, rel_l2_h,         
           color="#009E73",  
           label=r"$h$",            
           linewidth=1.2, linestyle="--",  zorder=3, marker="o", markersize=2,)

ax_2a.plot(noise, rel_l2_alpha,     
           color="#0072B2",  
           label=r"$\alpha$",       
           linewidth=1.2, linestyle="-", zorder=3, marker="s", markersize=2,)

ax_2a.plot(noise, rel_l2_h_dot,     
           color="#D55E00",  
           label=r"$\dot{h}$",      
           linewidth=1.2, linestyle="-.", zorder=3, marker="^", markersize=2,)

ax_2a.plot(noise, rel_l2_alpha_dot, 
           color="#4D4D4D", 
           label=r"$\dot{\alpha}$", 
           linewidth=1.2, linestyle=":",  zorder=3, marker="d", markersize=2,)
ax_2a.set_xlim([0,20])
ax_2a.set_xlabel(r"$\sigma (\%)$")
ax_2a.set_ylabel(r"Relative $L_2$ Error")
ax_2a.tick_params(direction="out", length=4, width=0.8)
ax_2a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_2a.xaxis.set_major_locator(MaxNLocator(4))
ax_2a.yaxis.set_major_locator(MaxNLocator(5))
ax_2a.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_2a.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_2a.tick_params(which="minor", direction="out", length=2, width=0.5)
ax_2a.grid(True,which="major",linestyle=":",linewidth=0.4,alpha=0.4)
for spine in ax_2a.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.show()

fig_2a.savefig("figures/Ex2_fig_2a.png", dpi=600, bbox_inches='tight')
fig_2a.savefig("figures/Ex2_fig_2a.pdf", bbox_inches="tight")

