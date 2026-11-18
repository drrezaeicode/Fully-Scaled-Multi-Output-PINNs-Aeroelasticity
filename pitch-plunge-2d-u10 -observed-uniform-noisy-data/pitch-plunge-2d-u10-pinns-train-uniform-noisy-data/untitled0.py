# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:19:26 2026

@author: Mohsen
"""

model.eval()

with torch.no_grad():
    u_pred_scaled_tensor = model(t_test_scaled_tensor)  # (N,2)
    u_pred_tensor = scaler.inverse_transformX(u_pred_scaled_tensor)

    fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    labels = [r"$h$", r"$\alpha$"]
    for i, ax in enumerate(axs):
        ax.plot(   t_exac_scaled_tensor.detach().cpu().numpy(), u_exac_scaled_tensor[:, i].detach().cpu().numpy(), label="Exact", color="blue"   , alpha=1)
        ax.scatter(t_data_scaled_tensor.detach().cpu().numpy(), u_data_scaled_tensor[:, i].detach().cpu().numpy(), label="Data" , color="black" , alpha=1, s=30)
        ax.plot(   t_test_scaled_tensor.detach().cpu().numpy(), u_pred_scaled_tensor[:, i].detach().cpu().numpy(), label="Prediction" , color="red" , alpha=1, linestyle="--")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(labels[i])
        ax.grid()
        ax.legend()
    
    fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    labels = [r"$h$", r"$\alpha$"]
    for i, ax in enumerate(axs):
        ax.plot(   t_exac_tensor.detach().cpu().numpy(), u_exac_tensor[:, i].detach().cpu().numpy(), label="Exact", color="blue"    , alpha=0.5)
        ax.scatter(t_data_tensor.detach().cpu().numpy(), u_data_tensor[:, i].detach().cpu().numpy(), label="Data" , color="black"  , alpha=1, s=30)
        ax.plot(   t_test_tensor.detach().cpu().numpy(), u_pred_tensor[:, i].detach().cpu().numpy(), label="Predicted" , color="red" , alpha=0.5, linestyle="--")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(labels[i])
        ax.grid()
        ax.legend()

plt.show()