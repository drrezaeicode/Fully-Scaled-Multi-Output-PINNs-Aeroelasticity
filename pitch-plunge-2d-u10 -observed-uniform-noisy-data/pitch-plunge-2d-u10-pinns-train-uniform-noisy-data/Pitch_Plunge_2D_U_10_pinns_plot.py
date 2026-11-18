from IPython import get_ipython
get_ipython().run_line_magic('clear','')
get_ipython().run_line_magic('reset' , '-f')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
device = "cpu"
import numpy as np
from Pitch_Plunge_2D_U_10_pinns_train import PINNs_MLPs  # اگر در ماژول جدا ذخیره شده
from save_load_function import load_globals
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.close("all")
import gc
gc.collect()           # آزاد کردن حافظه اشغال شده توسط figure های قدیمی


#loaded_workspace = load_globals("pitch_plunge_2D_U_10_train.pth", scope=globals())
loaded_workspace = load_globals("pitch_plunge_2D_U_10_pinns_train.pth", scope=globals())
#loaded_workspace = load_globals("pitch_plunge_2D_U_10_train_R1_data_only.pth", scope=globals())


model = PINNs_MLPs([1,128,128,2]).to(device)  # ورودی t، خروجی (H, α)

#model.load_state_dict(torch.load("mlp_pitch_plunge_2D_U_10_train.pth", map_location=device, weights_only=True))
model.load_state_dict(torch.load("pitch_plunge_2D_U_10_pinns_network.pth", map_location=device, weights_only=True))
#model.load_state_dict(torch.load("mlp_pitch_plunge_2D_U_10_train_R1_data_only.pth", map_location=device, weights_only=True))

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

ax_1a.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Exact}$"    ,color="#1f77b4",linewidth=1.5                ,zorder=0)
ax_1a.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Predicted}$",color="#d62728", linestyle="--",linewidth=1.2,zorder=1)
ax_1a.scatter(t_data_tensor.detach().cpu().numpy(),u_data_tensor[:, 0].detach().cpu().numpy(), label=r"$\mathrm{Data}$"    ,color="black"  , s=10,  marker='.',zorder=2)
ax_1a.scatter(t_ini_tensor.detach().cpu().numpy(), u_ini_tensor[:, 0].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)

ax_1a.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1a.set_ylabel(r"$h$")

ax_1a.tick_params(direction="in", length=4, width=0.8)
ax_1a.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

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

ax_1b.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Exact}$"    ,color="#1f77b4",linewidth=1.5                ,zorder=0)
ax_1b.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Predicted}$",color="#d62728", linestyle="--",linewidth=1.2,zorder=1)
ax_1b.scatter(t_data_tensor.detach().cpu().numpy(),u_data_tensor[:, 1].detach().cpu().numpy(), label=r"$\mathrm{Data}$"    ,color="black"  , s=10,  marker='.',zorder=2)
ax_1b.scatter(t_ini_tensor.detach().cpu().numpy(), u_ini_tensor[:, 1].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)

ax_1b.set_xlabel(r"$t\,(\mathrm{s})$")
ax_1b.set_ylabel(r"$\alpha$")

ax_1b.tick_params(direction="in", length=4, width=0.8)
ax_1b.legend(loc="lower center",bbox_to_anchor=(0.45, 1.02),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.2,handletextpad=0.5)

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

ax_1c.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Exact}$"    ,color="#1f77b4",linewidth=1.5                ,zorder=0)
ax_1c.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Predicted}$",color="#d62728", linestyle="--",linewidth=1.2,zorder=1)
ax_1c.scatter(t_ini_tensor.detach().cpu().numpy(), du_ini_tensor[:, 0].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)

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

ax_1d.plot(   t_exac_tensor.detach().cpu().numpy(),u_exac_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Exact}$"    ,color="#1f77b4",linewidth=1.5                ,zorder=0)
ax_1d.plot(   t_test_tensor.detach().cpu().numpy(),u_pred_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Predicted}$",color="#d62728", linestyle="--",linewidth=1.2,zorder=1)
ax_1d.scatter(t_ini_tensor.detach().cpu().numpy(), du_ini_tensor[:, 1].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)

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
fig_2a = plt.figure(figsize=(3.35, 2.7));ax_2a = fig_2a.add_subplot();fig_2a.subplots_adjust(left=0.18, bottom=0.18)

ax_2a.plot(  u_exac_tensor[:, 0].detach().cpu().numpy(),u_exac_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Exact}$"    ,color="#1f77b4",linewidth=1.5                ,zorder=0)
ax_2a.plot(  u_pred_tensor[:, 0].detach().cpu().numpy(),u_pred_dot_tensor[:, 0].detach().cpu().numpy(),label=r"$\mathrm{Predicted}$",color="#d62728", linestyle="--",linewidth=1.2,zorder=1)
ax_2a.scatter(u_ini_tensor[:,0].detach().cpu().numpy(),du_ini_tensor[:,0].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)

# he = u_exac_tensor[:, 0].detach().cpu().numpy()
# hed = u_exac_dot_tensor[:, 0].detach().cpu().numpy()
# hp = u_pred_tensor[:, 0].detach().cpu().numpy()
# hpd = u_pred_dot_tensor[:, 0].detach().cpu().numpy()
# n_arrow = 200
# eps = 1   # طول خیلی کوتاه فلش (مماس)
# for i in range(0, len(he) - eps, n_arrow):
#     ax_2a.annotate(
#         "",
#         xy=(he[i + eps], hed[i + eps]),
#         xytext=(he[i], hed[i]),
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
# for i in range(0, len(hp) - eps, n_arrow-1):
#     ax_2a.annotate(
#         "",
#         xy=(hp[i + eps], hpd[i + eps]),
#         xytext=(hp[i], hpd[i]),
#         arrowprops=dict(
#             arrowstyle="->",
#             color="#d62728",
#             lw=0.8,
#             mutation_scale=6,   # اندازه سر فلش
#             shrinkA=0,
#             shrinkB=0
#         ),
#         zorder=3
#     )    
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
fig_2b = plt.figure(figsize=(3.35, 2.7));ax_2b = fig_2b.add_subplot();fig_2b.subplots_adjust(left=0.18, bottom=0.18)

ax_2b.plot(  u_exac_tensor[:, 1].detach().cpu().numpy(),u_exac_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Exact}$"    ,color="#1f77b4",linewidth=1.5                ,zorder=0)
ax_2b.plot(  u_pred_tensor[:, 1].detach().cpu().numpy(),u_pred_dot_tensor[:, 1].detach().cpu().numpy(),label=r"$\mathrm{Predicted}$",color="#d62728", linestyle="--",linewidth=1.2,zorder=1)
ax_2b.scatter(u_ini_tensor[:, 1].detach().cpu().numpy(),du_ini_tensor[:,1].detach().cpu().numpy() , label=r"$\mathrm{Initial}$" ,color="#2ca02c", s=5,   marker='s',zorder=3)
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

fig_1a.savefig("figures/Ex2_fig_1a.png", dpi=600, bbox_inches='tight')
fig_1a.savefig("figures/Ex2_fig_1a.pdf", bbox_inches="tight")
fig_1b.savefig("figures/Ex2_fig_1b.png", dpi=600, bbox_inches='tight')
fig_1b.savefig("figures/Ex2_fig_1b.pdf", bbox_inches="tight")
fig_1c.savefig("figures/Ex2_fig_1c.png", dpi=600, bbox_inches='tight')
fig_1c.savefig("figures/Ex2_fig_1c.pdf", bbox_inches="tight")
fig_1d.savefig("figures/Ex2_fig_1d.png", dpi=600, bbox_inches='tight')
fig_1d.savefig("figures/Ex2_fig_1d.pdf", bbox_inches="tight")
fig_2a.savefig("figures/Ex2_fig_2a.png", dpi=600, bbox_inches='tight')
fig_2a.savefig("figures/Ex2_fig_2a.pdf", bbox_inches="tight")
fig_2b.savefig("figures/Ex2_fig_2b.png", dpi=600, bbox_inches='tight')
fig_2b.savefig("figures/Ex2_fig_2b.pdf", bbox_inches="tight")



'''


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.96,
                    wspace=0.15, hspace=0.25)
handles, labels = ax[0,0].get_legend_handles_labels()

fig.legend(handles, labels,
           loc='lower center',
           ncol=4,
           #frameon=True,
           #fancybox=True,
           #edgecolor='black',
           fontsize=14,
           bbox_to_anchor=(0.5, 0.05))   




fig1, ax1 = plt.subplots(1, 2, figsize=(24, 8))

ax1[0].plot(  u_exac_tensor[:, 0].detach().cpu().numpy(),u_exac_dot_tensor[:, 0].detach().cpu().numpy(), label="Exact", color="#1f77b4"    , alpha=1,linewidth=2.5,zorder=1 )
ax1[0].plot(  u_pred_tensor[:, 0].detach().cpu().numpy(),u_pred_dot_tensor[:, 0].detach().cpu().numpy(), label="Predicted" , color="#d62728" , alpha=1, linestyle="--",linewidth=2,zorder=2)
ax1[0].scatter(u_ini_tensor[:,0].detach().cpu().numpy(),du_ini_tensor[:,0].detach().cpu().numpy() , s=50, label= "Initial ", color = "#2ca02c", alpha = 1,zorder=4,marker='s')
ax1[0].set_xlabel(r"$h$")
ax1[0].set_ylabel(r"$\dot{h}$")
ax1[0].grid()
#ax1[0].legend()



ax1[1].plot(  u_exac_tensor[:, 1].detach().cpu().numpy(),u_exac_dot_tensor[:, 1].detach().cpu().numpy(), label="Exact", color="#1f77b4"    , alpha=1,linewidth=2.5,zorder=1 )
ax1[1].plot(  u_pred_tensor[:, 1].detach().cpu().numpy(),u_pred_dot_tensor[:, 1].detach().cpu().numpy(), label="Predicted" , color="#d62728" , alpha=1, linestyle="--",linewidth=2,zorder=2)
ax1[1].scatter(u_ini_tensor[:,1].detach().cpu().numpy(),du_ini_tensor[:,1].detach().cpu().numpy() , s=50, label= "Initial", color = "#2ca02c", alpha = 1,zorder=4,marker='s')
ax1[1].set_xlabel(r"$\alpha$")
ax1[1].set_ylabel(r"$\dot{\alpha}$")
ax1[1].grid()
#ax1[1].legend()



fig1.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.96,
                    wspace=0.15, hspace=0.25)

handles, labels = ax1[0].get_legend_handles_labels()
fig1.legend(handles, labels,
           loc='lower center',
           ncol=3,
           #frameon=True,
           #fancybox=True,
           #edgecolor='black',
           fontsize=14,
           bbox_to_anchor=(0.5, 0.04))   

plt.show()



#fig.savefig("figures/time_plot_R1_data_only.png", dpi=600, bbox_inches='tight')
#fig1.savefig( "figures/phase_space_U_10.png", dpi=600, bbox_inches='tight')

#fig.savefig("figures/time_plot_R1_data_only_R1.png", dpi=600, bbox_inches='tight')
#fig1.savefig( "figures/phase_space_U_10_R1.png", dpi=600, bbox_inches='tight')

#fig.savefig("figures/time_plot_R1_data_only_R1_data_only.png", dpi=600, bbox_inches='tight')
#fig1.savefig( "figures/phase_space_U_10_R1_data_only.png", dpi=600, bbox_inches='tight')






'''











