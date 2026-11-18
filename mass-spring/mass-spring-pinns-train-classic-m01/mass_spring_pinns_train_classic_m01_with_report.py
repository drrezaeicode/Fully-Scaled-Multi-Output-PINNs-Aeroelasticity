#%% PINNs for Nonlinear Problem

#%% 1. Import Necessary Libraries

# from IPython import get_ipython
# #get_ipython().run_line_magic('clear','')
# get_ipython().run_line_magic('reset' , '-f')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt; plt.close("all")
import datetime
import os
import time

plt.ioff()

import torch
import torch.nn as nn
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library
from save_load_function import save_globals
import random
seed = 42

# Python random
random.seed(seed)

# NumPy random
np.random.seed(seed)

# PyTorch CPU
torch.manual_seed(seed)
# PyTorch GPU
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'Using Device: {device}')




#%% Definge Class and Functions


def ode_system(t,Y,M_mat,K_mat,k3,e1,e2):
    y = Y[:2]
    y_dot = Y[2:]
    y_ddot = np.linalg.solve(M_mat,-( K_mat @ y + (e1-e2)*  k3 * ((y[0]-y[1])**3) )) # assuming y[1] = alpha
    return np.concatenate([y_dot, y_ddot])

def exact_solution(t,t_exact,u_exact):
    return np.array( [np.interp(t,t_exact.ravel(), u_exact[:,i].ravel()) for i in range(u_exact.shape[1])]).T



class PINNs_MLPs(nn.Module):
    def __init__(self,  layer_sizes, activation = torch.tanh,device=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.activation = activation

    def forward(self, x,activation=None):
        if activation is None:
            activation = self.activation
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x


def ode_loss(model, t, M_torch, K_torch, k3_torch, e1_torch, e2_torch):
    y = model(t)
    dy_dt  = torch.cat([
        torch.autograd.grad(y[:, i]    , t, grad_outputs=torch.ones_like(y[:, i])    , create_graph=True,retain_graph=True,only_inputs=True)[0]
        for i in range(y.shape[1])]     ,dim=1)
    d2y_dt2 = torch.cat([
        torch.autograd.grad(dy_dt[:, i], t, grad_outputs=torch.ones_like(dy_dt[:, i]), create_graph=True, retain_graph=True,only_inputs=True)[0]
        for i in range(dy_dt.shape[1])], dim=1)  # (N,2)
    
    nonlinearterm =((y[:, 0:1]-y[:, 1:2])**3).reshape(-1, 1) * (e1_torch.reshape(1, -1)-e2_torch.reshape(1, -1)) * k3_torch
    
    res = d2y_dt2 @ M_torch.T + y @ K_torch.T + nonlinearterm
    loss = torch.mean(res.pow(2), dim=0)
    return loss

def L2_relative_error(model, t_test_tensor, u_test_tensor):
    u_pred_tensor = model(t_test_tensor)
    err = torch.norm(u_pred_tensor - u_test_tensor)/torch.norm(u_test_tensor)
    return err

def data_loss(model, t_data_tensor, u_data_tensor):
    u_pred_tensor = model(t_data_tensor)  #output is scaled u
    loss = torch.mean((u_pred_tensor - u_data_tensor).pow(2),dim=0)
    return loss

def ic_loss(model, t, z0_true, weights=None):
    t = t.requires_grad_()
    z = model(t)  
    z0 = z[0, 0]
    z1 = z[0, 1]
    dz0 = torch.autograd.grad(z0, t, grad_outputs=torch.ones_like(z0), create_graph=True, retain_graph=True)[0]
    dz1 = torch.autograd.grad(z1, t, grad_outputs=torch.ones_like(z1), create_graph=True, retain_graph=True)[0]
    z_pred = torch.cat([z0.view(1), z1.view(1), dz0.view(1), dz1.view(1)], dim=0)
    loss_ic = (z_pred - z0_true).pow(2)
    return loss_ic










if __name__ == "__main__":
     
    #%%1. System parameters
    
    m1, m2, k1, k2, k3, k4       =  1,1,1, 1, 1,1
    z0       =[5,1,0,0]
    t_span=[0,6]
    exact_n, data_n,test_n,coll_n =6000, 15, 2000, 2000
    
    #%% 2. Main Matrix Data   
    M_mat = np.array([[m1,0],[0,m2]])
    M_inv = np.linalg.inv(M_mat)
    K_mat = np.array([[k1+k2,-k2],[-k2,k4+k2]])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    M_torch = torch.tensor(M_mat, dtype=torch.float32, device=device)
    K_torch = torch.tensor(K_mat, dtype=torch.float32, device=device)
    e1_torch    = torch.tensor(e1, dtype=torch.float32, device=device)
    e2_torch    = torch.tensor(e2, dtype=torch.float32, device=device)
    k3_torch = torch.tensor(k3, dtype=torch.float32, device=device)
    z0_tensor = torch.tensor(z0, dtype = torch.float32, device = device)
    
    #%% 3. Define Exact Solution of Real Equation
    
    
    
    # Exact
    t_exact = np.linspace(*t_span, exact_n, dtype=np.float64)
    sol= solve_ivp(ode_system,t_span,z0,args=(M_mat,K_mat,k3,e1,e2), t_eval=t_exact,method='BDF',rtol=1e-7,atol=1e-7)#
    t_exac = sol.t.reshape(-1,1)
    u_exac = sol.y.T[:,:2];
    t_exac_tensor = torch.tensor(t_exac, dtype = torch.float64, device = device)
    u_exac_tensor = torch.tensor(u_exac, dtype = torch.float64, device = device)
    
    # Data and Test
    t_data = np.linspace(*t_span, data_n, dtype=np.float64).reshape(-1,1);t_data = t_data[1:]
    t_test = np.linspace(*t_span, test_n, dtype=np.float64).reshape(-1,1)
    t_coll = np.linspace(*t_span, coll_n, dtype=np.float64).reshape(-1,1)
    t_init = np.array([[t_span[0]]])
    
    u_exac = sol.y.T[:,:2];
    u_exac_dot = sol.y.T[:,2:]
    u_data = exact_solution(t_data, t_exac, u_exac).squeeze(0)
    u_test = exact_solution(t_test, t_exac, u_exac).squeeze(0)
    u_ini  = np.array(z0[0:2],dtype=np.float32).reshape(1,-1) # 1 x 4 [h,alpha, h0, alpha0]
    du_ini = np.array(z0[2:4],dtype=np.float32).reshape(1,-1) # 1 x 4 [h,alpha, h0, alpha0]
    
    # ======================  to tensor ======================================
    t_exac_tensor = torch.tensor(t_exac, dtype = torch.float32, device = device)
    t_data_tensor = torch.tensor(t_data, dtype = torch.float32, device = device)
    t_test_tensor = torch.tensor(t_test, dtype = torch.float32, device = device)
    t_coll_tensor = torch.tensor(t_coll, dtype = torch.float32, device = device, requires_grad=True)
    t_ini_tensor  = torch.tensor(t_init, dtype = torch.float32, device = device)
    
    u_exac_dot_tensor = torch.tensor(u_exac_dot, dtype = torch.float32, device = device)
    u_exac_tensor = torch.tensor(u_exac, dtype = torch.float32, device = device)
    u_data_tensor = torch.tensor(u_data, dtype = torch.float32, device = device)
    u_test_tensor = torch.tensor(u_test, dtype = torch.float32, device = device)
    u_ini_tensor  = torch.tensor(u_ini , dtype = torch.float32, device = device)
    du_ini_tensor = torch.tensor(du_ini, dtype = torch.float32, device = device) 
    
  
    #%% 6. Train Network with exact data
    model = PINNs_MLPs([1,128,128,2]).to(device)  # ورودی t، خروجی (H, α)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimiser,
    milestones=[5000,10000,15000,20000,25000,30000,35000],
    gamma=(10**(-1))**(1/7))  
    loss_weights_tensor = torch.tensor([0.015, 110/14,3, 110], dtype = torch.float32, device = device)

    ODE_indices = [0,1]
    IC_indices = [2,3,4,5]
    Data_indices = [6,7]
    loss_weights_tensor /= loss_weights_tensor.sum()
    
    print("W:",loss_weights_tensor.detach().cpu().numpy())
    loss_history = []
    l2_history = []
    epoch_history = []
    lr_history = []
    log_every = 10
    n_adam = 60000
    if device.type == "cuda": torch.cuda.synchronize()
    t_start_adam = time.perf_counter()
    for i in range(n_adam):
        optimiser.zero_grad(set_to_none=True)
        current_lr      = optimiser.param_groups[0]["lr"]
        
        data     = data_loss(model, t_data_tensor, u_data_tensor)
        #print(t_data_tensor.shape,u_data_tensor.shape)  # should check
        ode      = ode_loss(model, t_coll_tensor,M_torch, K_torch, k3_torch, e1_torch, e2_torch)
        ic       = ic_loss(model, t_ini_tensor,z0_tensor)
    
    
        loss_raw        = torch.stack([ode[0]+ ode[1], ic[0]+ic[1], ic[2]+ ic[3], data[0]+ data[1]])
        loss_weighted   = loss_weights_tensor * loss_raw
        loss_total      = torch.sum(loss_weighted)
        
        

        
        if i % log_every == 0:                       # monitoring (not every epoch, to keep the timing clean)
            with torch.no_grad():
                l2er = L2_relative_error(model, t_test_tensor, u_test_tensor)
            loss_history.append(loss_total.item())
            l2_history.append(l2er.item())
            epoch_history.append(i)
            lr_history.append(current_lr)
        loss_total.backward()
        optimiser.step()
        scheduler.step()
        

        
        
        if i % 1000 == 0:
            print(
                f"Epoch {i:5d}, Loss: {loss_total.item():.4f}, LR: {current_lr:.2e}, "
                f"L2E: {l2er.item():.4f}, RLoss: [{', '.join(f'{x:.4e}' for x in loss_raw.detach().cpu().numpy())}]"
            )
    # Sure Network with much Data Work


    if device.type == "cuda": torch.cuda.synchronize()
    time_adam = time.perf_counter() - t_start_adam
    model.eval()

    #%% Final report: test errors (physical units) and computational cost
    if device.type == "cuda": torch.cuda.synchronize()
    t_te = t_test_tensor.clone().requires_grad_(True)
    u_s  = model(t_te)
    du_s = torch.cat([torch.autograd.grad(u_s[:, i:i+1], t_te,
                      grad_outputs=torch.ones_like(u_s[:, i:i+1]), retain_graph=True)[0]
                      for i in range(2)], dim=1)
    u_pred  = u_s.detach().cpu().numpy()
    du_pred = du_s.detach().cpu().numpy()

    u_ref  = np.asarray(u_test).reshape(-1, 2)                               # [x1, x2]
    du_ref = np.stack([np.interp(t_test.ravel(), t_exac.ravel(), u_exac_dot[:, k]) for k in range(2)], axis=1)

    def rel_l2(p, r):  return np.linalg.norm(p - r, axis=0) / np.linalg.norm(r, axis=0)
    def rel_max(p, r): return np.max(np.abs(p - r), axis=0) / np.max(np.abs(r), axis=0)

    names   = ["x1", "x2", "x1dot", "x2dot"]
    err_l2  = 100 * np.concatenate([rel_l2(u_pred, u_ref),  rel_l2(du_pred, du_ref)])
    err_max = 100 * np.concatenate([rel_max(u_pred, u_ref), rel_max(du_pred, du_ref)])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    lw = loss_weights_tensor.detach().cpu().numpy()

    lines = []
    lines.append("=" * 72)
    lines.append("REPORT: classic_m01  |  masses m1={}, m2={}".format(m1, m2))
    lines.append("=" * 72)
    lines.append("[Setup]")
    lines.append(f"  Device                      : {dev_name}")
    lines.append(f"  Network                     : [1,128,128,2], trainable parameters = {n_params}")
    lines.append(f"  Data points N_d             : {t_data.shape[0]}")
    lines.append(f"  Collocation points N_f      : {t_coll.shape[0]}")
    lines.append(f"  Test points N_test          : {t_test.shape[0]}")
    lines.append(f"  Normalised loss weights     : ode={lw[0]:.4e}, ic_disp={lw[1]:.4e}, ic_vel={lw[2]:.4e}, data={lw[3]:.4e}")
    lines.append(f"  Adam epochs                 : {n_adam} (initial lr = 1e-3)")
    lines.append("[Computational cost]")
    lines.append(f"  Training time (Adam)        : {time_adam:10.1f} s  ({time_adam/60:.1f} min)")
    lines.append(f"  Time per epoch              : {1000*time_adam/n_adam:10.2f} ms")
    lines.append("[Test errors, physical units, in %]")
    lines.append(f"  {'state':10s}{'rel. L2 error':>16s}{'max err/max|ref|':>20s}")
    for n_, a_, b_ in zip(names, err_l2, err_max):
        lines.append(f"  {n_:10s}{a_:16.3e}{b_:20.3e}")
    lines.append("=" * 72)
    report_text = "\n".join(lines)
    print(report_text)

    with open("report_classic_m01.txt", "w", encoding="utf-8") as f_:
        f_.write(report_text + "\n")
    with open("report_classic_m01.csv", "w", encoding="utf-8") as f_:
        f_.write("state,rel_L2_percent,max_over_maxref_percent\n")
        for n_, a_, b_ in zip(names, err_l2, err_max):
            f_.write(f"{n_},{a_:.6e},{b_:.6e}\n")
        f_.write(f"time_adam_s,{time_adam:.2f},\nparams,{n_params},\n")

    
    with torch.no_grad():
        u_pred_tensor = model(t_test_tensor)  # (N,2)
    
        fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
        labels = [r"$h$", r"$\alpha$"]
        for i, ax in enumerate(axs):
            ax.plot(   t_exac_tensor.detach().cpu().numpy(), u_exac_tensor[:, i].detach().cpu().numpy(), label="Exact", color="blue"   , alpha=1)
            ax.scatter(t_data_tensor.detach().cpu().numpy(), u_data_tensor[:, i].detach().cpu().numpy(), label="Data" , color="black" , alpha=1, s=30)
            ax.plot(   t_test_tensor.detach().cpu().numpy(), u_pred_tensor[:, i].detach().cpu().numpy(), label="Prediction" , color="red" , alpha=1, linestyle="--")
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

    #plt.show()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"figures/plot_{timestamp}_.png")
    
    #print(best_params)
    plt.figure(figsize=(7,5))
    plt.semilogy(epoch_history, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.figure(figsize=(7,5))
    plt.semilogy(epoch_history, l2_history)
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.grid(True)
        
    #plt.show()
    print(loss_raw)
    print(loss_weighted)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f"figures/figure_{i}_{timestamp}.png", dpi=300)
    
    plt.show()
    

    # save_globals("mass-spring-pinns-train-classic-m01.pth", scope=globals())
    # torch.save(model.state_dict(), "mass-spring-pinns-train-classic-m01_network.pth")

    
    
    

    #os.system("shutdown /h")