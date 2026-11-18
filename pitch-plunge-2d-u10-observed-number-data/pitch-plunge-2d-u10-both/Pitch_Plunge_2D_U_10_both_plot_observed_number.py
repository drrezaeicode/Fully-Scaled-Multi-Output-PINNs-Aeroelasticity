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
sys.path.append(os.path.abspath("../pitch-plunge-2d-u10-pinns-train-observed"))


from Pitch_Plunge_2D_U_10_pinns_train_40 import PINNs_MLPs 



from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_30.pth", scope=globals())
model = PINNs_MLPs([1,128,128,2]).to(device) 
model.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_30.pth", map_location=device, weights_only=True))
model.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
t_test_scaled_tensor_30 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_30 = model(t_test_scaled_tensor_30)  # (N,2)
u_pred_dot_scaled_tensor_30 = torch.zeros_like(u_pred_scaled_tensor_30)
u_pred_dot_scaled_tensor_30[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_30[:, 0:1], t_test_scaled_tensor_30, grad_outputs=torch.ones_like(u_pred_scaled_tensor_30[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_30[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_30[:, 1:2], t_test_scaled_tensor_30, grad_outputs=torch.ones_like(u_pred_scaled_tensor_30[:, 0:1]), create_graph=True)[0]
u_pred_tensor_30 = scaler.inverse_transformX(u_pred_scaled_tensor_30)
u_pred_dot_tensor_30 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_30)
t_data_tensor_30 = t_data_tensor
u_data_tensor_30 = u_data_tensor



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_40.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_40.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_40 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_40 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_40 = torch.zeros_like(u_pred_scaled_tensor_40)
u_pred_dot_scaled_tensor_40[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_40[:, 0:1], t_test_scaled_tensor_40, grad_outputs=torch.ones_like(u_pred_scaled_tensor_40[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_40[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_40[:, 1:2], t_test_scaled_tensor_40, grad_outputs=torch.ones_like(u_pred_scaled_tensor_40[:, 0:1]), create_graph=True)[0]
u_pred_tensor_40 = scaler.inverse_transformX(u_pred_scaled_tensor_40)
u_pred_dot_tensor_40 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_40)
t_data_tensor_40 = t_data_tensor
u_data_tensor_40 = u_data_tensor


loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_50.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_50.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_50 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_50 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_50 = torch.zeros_like(u_pred_scaled_tensor_50)
u_pred_dot_scaled_tensor_50[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_50[:, 0:1], t_test_scaled_tensor_50, grad_outputs=torch.ones_like(u_pred_scaled_tensor_50[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_50[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_50[:, 1:2], t_test_scaled_tensor_50, grad_outputs=torch.ones_like(u_pred_scaled_tensor_50[:, 0:1]), create_graph=True)[0]
u_pred_tensor_50 = scaler.inverse_transformX(u_pred_scaled_tensor_50)
u_pred_dot_tensor_50 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_50)
t_data_tensor_50 = t_data_tensor
u_data_tensor_50 = u_data_tensor

loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_20.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_20.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_20 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_20 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_20 = torch.zeros_like(u_pred_scaled_tensor_20)
u_pred_dot_scaled_tensor_20[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_20[:, 0:1], t_test_scaled_tensor_20, grad_outputs=torch.ones_like(u_pred_scaled_tensor_20[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_20[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_20[:, 1:2], t_test_scaled_tensor_20, grad_outputs=torch.ones_like(u_pred_scaled_tensor_20[:, 0:1]), create_graph=True)[0]
u_pred_tensor_20 = scaler.inverse_transformX(u_pred_scaled_tensor_20)
u_pred_dot_tensor_20 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_20)
t_data_tensor_20 = t_data_tensor
u_data_tensor_20 = u_data_tensor



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_10.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_10.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_10 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_10 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_10 = torch.zeros_like(u_pred_scaled_tensor_10)
u_pred_dot_scaled_tensor_10[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_10[:, 0:1], t_test_scaled_tensor_10, grad_outputs=torch.ones_like(u_pred_scaled_tensor_10[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_10[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_10[:, 1:2], t_test_scaled_tensor_10, grad_outputs=torch.ones_like(u_pred_scaled_tensor_10[:, 0:1]), create_graph=True)[0]
u_pred_tensor_10 = scaler.inverse_transformX(u_pred_scaled_tensor_10)
u_pred_dot_tensor_10 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_10)
t_data_tensor_10 = t_data_tensor
u_data_tensor_10 = u_data_tensor



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_15.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_15.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_15 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_15 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_15 = torch.zeros_like(u_pred_scaled_tensor_15)
u_pred_dot_scaled_tensor_15[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_15[:, 0:1], t_test_scaled_tensor_15, grad_outputs=torch.ones_like(u_pred_scaled_tensor_15[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_15[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_15[:, 1:2], t_test_scaled_tensor_15, grad_outputs=torch.ones_like(u_pred_scaled_tensor_15[:, 0:1]), create_graph=True)[0]
u_pred_tensor_15 = scaler.inverse_transformX(u_pred_scaled_tensor_15)
u_pred_dot_tensor_15 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_15)
t_data_tensor_15 = t_data_tensor
u_data_tensor_15 = u_data_tensor



loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_25.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_25.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_25 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_25 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_25 = torch.zeros_like(u_pred_scaled_tensor_25)
u_pred_dot_scaled_tensor_25[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_25[:, 0:1], t_test_scaled_tensor_25, grad_outputs=torch.ones_like(u_pred_scaled_tensor_25[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_25[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_25[:, 1:2], t_test_scaled_tensor_25, grad_outputs=torch.ones_like(u_pred_scaled_tensor_25[:, 0:1]), create_graph=True)[0]
u_pred_tensor_25 = scaler.inverse_transformX(u_pred_scaled_tensor_25)
u_pred_dot_tensor_25 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_25)
t_data_tensor_25 = t_data_tensor
u_data_tensor_25 = u_data_tensor


loaded_workspace = load_globals("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_train_60.pth", scope=globals())
model_d = PINNs_MLPs([1,128,128,2]).to(device) 
model_d.load_state_dict(torch.load("../pitch-plunge-2d-u10-pinns-train-observed/pitch_plunge_2D_U_10_pinns_network_60.pth", map_location=device, weights_only=True))
model_d.eval()
scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
# obtain t_test and Prediction Data
t_test_scaled_tensor_60 = t_test_scaled_tensor.requires_grad_(True)
u_pred_scaled_tensor_60 = model_d(t_test_scaled_tensor)  # (N,2)
u_pred_dot_scaled_tensor_60 = torch.zeros_like(u_pred_scaled_tensor_60)
u_pred_dot_scaled_tensor_60[:,0:1] = torch.autograd.grad(u_pred_scaled_tensor_60[:, 0:1], t_test_scaled_tensor_60, grad_outputs=torch.ones_like(u_pred_scaled_tensor_60[:, 0:1]), create_graph=True)[0]
u_pred_dot_scaled_tensor_60[:,1:2] = torch.autograd.grad(u_pred_scaled_tensor_60[:, 1:2], t_test_scaled_tensor_60, grad_outputs=torch.ones_like(u_pred_scaled_tensor_60[:, 0:1]), create_graph=True)[0]
u_pred_tensor_60 = scaler.inverse_transformX(u_pred_scaled_tensor_60)
u_pred_dot_tensor_60 = scaler.scale_first_derivative_inverse_transform(u_pred_dot_scaled_tensor_60)
t_data_tensor_60 = t_data_tensor
u_data_tensor_60 = u_data_tensor




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
error_10     = u_exac_tensor - u_pred_tensor_10
error_20     = u_exac_tensor - u_pred_tensor_20
error_30     = u_exac_tensor - u_pred_tensor_30
error_40     = u_exac_tensor - u_pred_tensor_40
error_50     = u_exac_tensor - u_pred_tensor_50

error_dot_10 = u_exac_dot_tensor - u_pred_dot_tensor_10
error_dot_20 = u_exac_dot_tensor - u_pred_dot_tensor_20
error_dot_30 = u_exac_dot_tensor - u_pred_dot_tensor_30
error_dot_40 = u_exac_dot_tensor - u_pred_dot_tensor_40
error_dot_50 = u_exac_dot_tensor - u_pred_dot_tensor_50





# Example 0 : Figure 1.a  ==================================================================================================
fig_1a = plt.figure(figsize=(3.35, 2.7));ax_1a = fig_1a.add_subplot();fig_1a.subplots_adjust(left=0.18, bottom=0.18)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_10[:, 0].detach().cpu().numpy(),label=r"$n_d = 10$",color="yellow",linewidth=1.0,linestyle="-",zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_20[:, 0].detach().cpu().numpy(),label=r"$n_d = 20$",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_30[:, 0].detach().cpu().numpy(),label=r"$n_d = 30$",color="green",linewidth=1.0,linestyle=":",zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_40[:, 0].detach().cpu().numpy(),label=r"$n_d = 40$",color="black",linewidth=1.0,linestyle="-.", zorder=1)
ax_1a.plot(t_test_tensor.detach().cpu().numpy(),error_50[:, 0].detach().cpu().numpy(),label=r"$n_d = 50$",color="red",linewidth=1.0,linestyle="--", zorder=1)
ax_1a.set_xlim([0,2])
ax_1a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1a.set_ylabel(r"$e_h$")
ax_1a.tick_params(direction="in", length=4, width=0.8)
ax_1a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=1,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_10[:, 1].detach().cpu().numpy(),label=r"$n_d = 20$",color="yellow",linewidth=1.0,linestyle="-",zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_20[:, 1].detach().cpu().numpy(),label=r"$n_d = 20$",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_30[:, 1].detach().cpu().numpy(),label=r"$n_d = 30$",color="green",linewidth=1.0,linestyle=":",zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_40[:, 1].detach().cpu().numpy(),label=r"$n_d = 40$",color="black",linewidth=1.0,linestyle="-.", zorder=1)
ax_1b.plot(t_test_tensor.detach().cpu().numpy(),error_50[:, 1].detach().cpu().numpy(),label=r"$n_d = 50$",color="red",linewidth=1.0,linestyle="--", zorder=1)
ax_1b.set_xlim([0,2])
ax_1b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1b.set_ylabel(r"$e_\alpha$")
ax_1b.tick_params(direction="in", length=4, width=0.8)
ax_1b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=1,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_dot_10[:, 0].detach().cpu().numpy(),label=r"$n_d = 10",color="yellow",linewidth=1.0,linestyle="-",zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_dot_20[:, 0].detach().cpu().numpy(),label=r"$n_d = 20",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_dot_30[:, 0].detach().cpu().numpy(),label=r"$n_d = 30",color="green",linewidth=1.0,linestyle=":",zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_dot_40[:, 0].detach().cpu().numpy(),label=r"$n_d = 40",color="black",linewidth=1.0,linestyle="-.", zorder=1)
ax_1c.plot(t_test_tensor.detach().cpu().numpy(),error_dot_50[:, 0].detach().cpu().numpy(),label=r"$n_d = 50",color="red",linewidth=1.0,linestyle="--", zorder=1)
ax_1c.set_xlim([0,2])
ax_1c.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1c.set_ylabel(r"$e_\dot{h}$")
ax_1c.tick_params(direction="in", length=4, width=0.8)
ax_1c.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=1,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_dot_10[:, 1].detach().cpu().numpy(),label=r"$n_d = 10$",color="yellow",linewidth=1.0,linestyle="-",zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_dot_20[:, 1].detach().cpu().numpy(),label=r"$n_d = 20$",color="#1E90FF",linewidth=1.0,linestyle="-",zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_dot_30[:, 1].detach().cpu().numpy(),label=r"$n_d = 30$",color="green",linewidth=1.0,linestyle=":",zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_dot_40[:, 1].detach().cpu().numpy(),label=r"$n_d = 40$",color="black",linewidth=1.0,linestyle="-.", zorder=1)
ax_1d.plot(t_test_tensor.detach().cpu().numpy(),error_dot_50[:, 1].detach().cpu().numpy(),label=r"$n_d = 50$",color="red",linewidth=1.0,linestyle="--", zorder=1)
ax_1d.set_xlim([0,2])
ax_1d.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1d.set_ylabel(r"$e_\dot{\alpha}$")
ax_1d.tick_params(direction="in", length=4, width=0.8)
ax_1d.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
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

n_observed_data = [10, 15, 20, 25, 30, 40, 50, 60]


u_pred_dict = {
    10: u_pred_tensor_10,
    15: u_pred_tensor_15,
    20: u_pred_tensor_20,
    25: u_pred_tensor_25,
    30: u_pred_tensor_30,
    40: u_pred_tensor_40,
    50: u_pred_tensor_50,
    60: u_pred_tensor_60,
}

u_pred_dot_dict = {
    10: u_pred_dot_tensor_10,
    15: u_pred_dot_tensor_15,
    20: u_pred_dot_tensor_20,
    25: u_pred_dot_tensor_25,
    30: u_pred_dot_tensor_30,
    40: u_pred_dot_tensor_40,
    50: u_pred_dot_tensor_50,
    60: u_pred_dot_tensor_60,
}


mse_list = []
mae_list = []
rmse_list = []

mse_dot_list = []
mae_dot_list = []
rmse_dot_list = []

for n in n_observed_data:

    mse = ((u_exac_tensor - u_pred_dict[n])**2).mean(dim=0)
    mae = torch.abs(u_exac_tensor - u_pred_dict[n]).mean(dim=0)
    rmse = torch.sqrt(mse)

    mse_dot = ((u_exac_dot_tensor - u_pred_dot_dict[n])**2).mean(dim=0)
    mae_dot = torch.abs(u_exac_dot_tensor - u_pred_dot_dict[n]).mean(dim=0)
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
mse_h, mse_alpha = mse_arr[:,0], mse_arr[:,1]
mae_h, mae_alpha = mae_arr[:,0], mae_arr[:,1]
rmse_h, rmse_alpha = rmse_arr[:,0], rmse_arr[:,1]

mse_h_dot, mse_alpha_dot = mse_dot_arr[:,0], mse_dot_arr[:,1]
mae_h_dot, mae_alpha_dot = mae_dot_arr[:,0], mae_dot_arr[:,1]
rmse_h_dot, rmse_alpha_dot = rmse_dot_arr[:,0], rmse_dot_arr[:,1]


rel_l2_list = []
rel_l2_dot_list = []

for n in n_observed_data:

    rel_l2 = (
        torch.linalg.norm(u_exac_tensor - u_pred_dict[n], dim=0)
        / torch.linalg.norm(u_exac_tensor, dim=0)
    )

    rel_l2_dot = (
        torch.linalg.norm(u_exac_dot_tensor - u_pred_dot_dict[n], dim=0)
        / torch.linalg.norm(u_exac_dot_tensor, dim=0)
    )

    rel_l2_list.append(rel_l2)
    rel_l2_dot_list.append(rel_l2_dot)

# shape = (N_cases, 2)
rel_l2_arr = torch.stack(rel_l2_list).detach().cpu().numpy()
rel_l2_dot_arr = torch.stack(rel_l2_dot_list).detach().cpu().numpy()

# Separate h and alpha
rel_l2_h         = rel_l2_arr[:, 0]
rel_l2_alpha     = rel_l2_arr[:, 1]

rel_l2_h_dot     = rel_l2_dot_arr[:, 0]
rel_l2_alpha_dot = rel_l2_dot_arr[:, 1]


n_observed_data = np.array([10, 15, 20, 25, 30, 40, 50, 60])


# Example 0 : Figure 2.a  ==================================================================================================
fig_2a = plt.figure(figsize=(3.35, 2.7));ax_2a = fig_2a.add_subplot();fig_2a.subplots_adjust(left=0.18, bottom=0.18)
ax_2a.plot(n_observed_data, rel_l2_h,         
           color="#009E73",  
           label=r"$h$",            
           linewidth=1.2, linestyle="--",  zorder=3, marker="o", markersize=2,)

ax_2a.plot(n_observed_data, rel_l2_alpha,     
           color="#0072B2",  
           label=r"$\alpha$",       
           linewidth=1.2, linestyle="-", zorder=3, marker="s", markersize=2,)

ax_2a.plot(n_observed_data, rel_l2_h_dot,     
           color="#D55E00",  
           label=r"$\dot{h}$",      
           linewidth=1.2, linestyle="-.", zorder=3, marker="^", markersize=2,)

ax_2a.plot(n_observed_data, rel_l2_alpha_dot, 
           color="#4D4D4D", 
           label=r"$\dot{\alpha}$", 
           linewidth=1.2, linestyle=":",  zorder=3, marker="d", markersize=2,)
ax_2a.set_xlim([10,60])
ax_2a.set_xlabel("Number of observed data")
ax_2a.set_ylabel(r"Relative $L_2$ Error")
ax_2a.tick_params(direction="out", length=4, width=0.8)
ax_2a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)
ax_2a.xaxis.set_major_locator(MaxNLocator(5))
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