#%% PINNs for Nonlinear Problem

#%% 1. Import Necessary Libraries

from IPython import get_ipython
get_ipython().run_line_magic('clear','')
get_ipython().run_line_magic('reset' , '-f')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt; plt.close("all")
import datetime
import os

plt.ioff()

import torch
import torch.nn as nn
from torch_data_scale import MinMaxScalerPytorchTX as mmsptx #My Library
from save_load_function import save_globals
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#torch.backends.cudnn.benchmark=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f'Using Device: {device}')




#%% Definge Class and Functions


def ode_system(t,Y,M_mat,K_mat,k3,e1,e2):
    y = Y[:2]
    y_dot = Y[2:]
    y_ddot = np.linalg.solve(M_mat,-( K_mat @ y + (e1-e2)*  k3 * ((y[0]-y[1])**3) )) # assuming y[1] = alpha
    return np.concatenate([y_dot, y_ddot])

def exact_solution(t,t_exact,u_exact):
    return np.array( [np.interp(t,t_exact.ravel(), u_exact[:,i].ravel()) for i in range(u_exact.shape[1])]).T



def ode_system_scaled(t, Y, M_bar, K_bar, K_0, k3, e1, e2, sfh, smh, sfa, sma):
    y = Y[:2]
    y_dot = Y[2:]
    nonlinear_term = k3 * ((y[0]-smh)/sfh-(y[1]-sma)/sfa)**3
    y_ddot = np.linalg.solve(M_bar,-( K_bar @ y + (e1-e2)* nonlinear_term - K_0  ) ) # assuming y[1] = alpha   
    return np.concatenate([y_dot, y_ddot])


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


def ode_loss(model, t, M_bar_torch, K_bar_torch, K_0_torch, k3_torch, e1_torch, e2_torch, sfh_tensor,smh_tensor, sfa_tensor,sma_tensor):
    y = model(t)
    dy_dt  = torch.cat([
        torch.autograd.grad(y[:, i]    , t, grad_outputs=torch.ones_like(y[:, i])    , create_graph=True,retain_graph=True,only_inputs=True)[0]
        for i in range(y.shape[1])]     ,dim=1)
    d2y_dt2 = torch.cat([
        torch.autograd.grad(dy_dt[:, i], t, grad_outputs=torch.ones_like(dy_dt[:, i]), create_graph=True, retain_graph=True,only_inputs=True)[0]
        for i in range(dy_dt.shape[1])], dim=1)  # (N,2)
    
    nonlinearterm =(((y[:, 0:1]-smh_tensor)/sfh_tensor-(y[:, 1:2]-sma_tensor)/sfa_tensor)**3).reshape(-1, 1) * (e1_torch.reshape(1, -1)-e2_torch.reshape(1, -1)) * k3_torch
    
    res = d2y_dt2 @ M_bar_torch.T + y @ K_bar_torch.T-K_0_torch + nonlinearterm
    loss = torch.mean(res.pow(2), dim=0)
    return loss

def L2_relative_error(model, t_test_scaled_tensor, u_test_scaled_tensor):
    u_pred_scaled_tensor = model(t_test_scaled_tensor)
    err = torch.norm(u_pred_scaled_tensor - u_test_scaled_tensor)/torch.norm(u_test_scaled_tensor)
    return err

def data_loss(model, t_data_scaled_tensor, u_data_scaled_tensor):
    u_pred_scaled_tensor = model(t_data_scaled_tensor)  #output is scaled u
    loss = torch.mean((u_pred_scaled_tensor - u_data_scaled_tensor).pow(2),dim=0)
    return loss

def ic_loss(model, t, z0_scaled, weights=None):
    t = t.requires_grad_()
    z = model(t)  
    z0 = z[0, 0]
    z1 = z[0, 1]
    dz0 = torch.autograd.grad(z0, t, grad_outputs=torch.ones_like(z0), create_graph=True, retain_graph=True)[0]
    dz1 = torch.autograd.grad(z1, t, grad_outputs=torch.ones_like(z1), create_graph=True, retain_graph=True)[0]
    z_pred = torch.cat([z0.view(1), z1.view(1), dz0.view(1), dz1.view(1)], dim=0)
    loss_ic = (z_pred - z0_scaled).pow(2)
    return loss_ic










if __name__ == "__main__":
     
    #%%1. System parameters
    
    m, k1, k2, k3       =  1, 1, 1, 1
    z0       =[5,1,0,0]
    t_span=[0,6]
    exact_n, data_n,test_n,coll_n =6000, 15, 2000, 1000
    
    #%% 2. Main Matrix Data   
    M_mat = np.array([[m,0],[0,m]])
    M_inv = np.linalg.inv(M_mat)
    K_mat = np.array([[k1+k2,-k2],[-k2,k1+k2]])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    
    
    #%% 3. Define Exact Solution of Real Equation
    
    
    
    # Exact
    t_exact = np.linspace(*t_span, exact_n, dtype=np.float64)
    sol= solve_ivp(ode_system,t_span,z0,args=(M_mat,K_mat,k3,e1,e2), t_eval=t_exact,method='BDF',rtol=1e-7,atol=1e-7)#
    t_exac = sol.t.reshape(-1,1)
    u_exac = sol.y.T[:,:2];
    t_exac_tensor = torch.tensor(t_exac, dtype = torch.float64, device = device)
    u_exac_tensor = torch.tensor(u_exac, dtype = torch.float64, device = device)
    
    # Data and Test
    t_data = np.linspace(*t_span, data_n, dtype=np.float64).reshape(-1,1);#t_data = t_data[1:]
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
    t_coll_tensor = torch.tensor(t_coll, dtype = torch.float32, device = device)
    t_ini_tensor  = torch.tensor(t_init, dtype = torch.float32, device = device)
    
    u_exac_dot_tensor = torch.tensor(u_exac_dot, dtype = torch.float32, device = device)
    u_exac_tensor = torch.tensor(u_exac, dtype = torch.float32, device = device)
    u_data_tensor = torch.tensor(u_data, dtype = torch.float32, device = device)
    u_test_tensor = torch.tensor(u_test, dtype = torch.float32, device = device)
    u_ini_tensor  = torch.tensor(u_ini , dtype = torch.float32, device = device)
    du_ini_tensor = torch.tensor(du_ini, dtype = torch.float32, device = device)
    
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))#, sharex=True
    labels = [r"$h$", r"$\alpha$"]
    for i, ax in enumerate(axs):
        ax.plot(   t_exac, u_exac[:, i],       label="Exact", color="black"  , alpha=0.5)
        ax.scatter(t_data, u_data[:, i], s=30, label="Data", color="tab:red" , alpha=1  )
        ax.scatter(t_test, u_test[:, i], s=10, label="Test", color="tab:blue", alpha=0.7)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(labels[i])
        ax.grid()
        ax.legend()
    
    plt.show()
    
    #%% 4. Define Exact Solution of Scaled Equation and Check Scaled equation
    
    scaler = mmsptx(t_range=(-1,1),x_range=(-1,1),device=device).fit(t_exac_tensor, u_exac_tensor)
    sfh = scaler.xscale_[0,0].item()
    smh = scaler.xmin_[0,0].item()
    sfa = scaler.xscale_[0,1].item()
    sma = scaler.xmin_[0,1].item()
    sft = scaler.tscale_.item()
    smt = scaler.tmin_.item()

    
    
    z0_scaled = [z0[0]*sfh+smh,z0[1]*sfa+sma,sft/sfh*z0[2],sft/sfa*z0[3]]
    z0_scaled_tensor = torch.tensor(z0_scaled, dtype = torch.float32, device = device)
    
    DK = np.diag((1 / scaler.xscale_).cpu().numpy().flatten())
    DM = np.diag((scaler.tscale_**2 / scaler.xscale_).cpu().numpy().flatten())# * scaler.tscale_.item()
    D0 = (scaler.xmin_/scaler.xscale_).detach().cpu().numpy()
    
    M_bar = M_mat@DM
    print(M_bar)
    M_bar = sft**2*M_mat @ np.linalg.inv(np.diag(scaler.xscale_.cpu().numpy().flatten()))
    print(M_bar)
    K_bar = K_mat@DK
    K_0   = K_mat@D0.ravel()
    
    
    
    
    t_exac_scaled = scaler.transformT(t_exac_tensor).detach().cpu().numpy().ravel()
    sol= solve_ivp(ode_system_scaled,[float(t_exac_scaled[0]),float(t_exac_scaled[-1])],z0_scaled,args=(M_bar, K_bar, K_0, k3, e1, e2, sfh, smh, sfa, sma), t_eval=t_exac_scaled,method='BDF',rtol=1e-7,atol=1e-7)#
    t_exac_scaled = sol.t.reshape(-1,1)
    u_exac_scaled = sol.y.T[:,:2];
    u_exac_dot_scaled=sol.y.T[:,2:]    
    
    fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    labels = [r"$h$", r"$\alpha$"]
    for i, ax in enumerate(axs):
        ax.plot(   t_exac_scaled, u_exac_scaled[:, i],       label="Exact", color="black"  , alpha=0.5)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(labels[i])
        ax.grid()
        ax.legend()
    
    plt.show()
    
    
    
    
    sfh_tensor = torch.tensor(sfh, dtype = torch.float32, device = device)
    smh_tensor = torch.tensor(smh, dtype = torch.float32, device = device)
    sfa_tensor = torch.tensor(sfa, dtype = torch.float32, device = device)
    sma_tensor = torch.tensor(sma, dtype = torch.float32, device = device)    
    
    M_bar_torch = torch.tensor(M_bar, dtype=torch.float32, device=device)
    K_bar_torch = torch.tensor(K_bar, dtype=torch.float32, device=device)
    e1_torch    = torch.tensor(e1, dtype=torch.float32, device=device)
    e2_torch    = torch.tensor(e2, dtype=torch.float32, device=device)
    k3_torch = torch.tensor(k3, dtype=torch.float32, device=device)
    K_0_torch = torch.tensor(K_0, dtype = torch.float32, device = device)
    
    #%% 5. Define MLP
    
    
    t_exac_scaled_tensor     = torch.tensor(t_exac_scaled, dtype = torch.float32, device = device)
    u_exac_scaled_tensor     = torch.tensor(u_exac_scaled, dtype = torch.float32, device = device)
    u_exac_dot_scaled_tensor = torch.tensor(u_exac_dot_scaled, dtype = torch.float32, device = device)
    
    
    t_data_scaled_tensor, u_data_scaled_tensor = scaler.transform(t_data_tensor, u_data_tensor)
    t_test_scaled_tensor, u_test_scaled_tensor = scaler.transform(t_test_tensor, u_test_tensor)
    t_ini_scaled_tensor , u_ini_scaled_tensor  = scaler.transform(t_ini_tensor , u_ini_tensor);t_ini_scaled_tensor =t_ini_scaled_tensor.requires_grad_(True)
    du_ini_scaled_tensor                       = scaler.scale_first_derivative_transform(du_ini_tensor)
    t_coll_scaled_tensor                       = scaler.transformT(t_coll_tensor);t_coll_scaled_tensor=t_coll_scaled_tensor.requires_grad_(True)
    z00_scaled_tensor                           = torch.cat([u_ini_scaled_tensor, du_ini_scaled_tensor],dim=1)
    
    
    fig, axs = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    labels = [r"$h$", r"$\alpha$"]
    for i, ax in enumerate(axs):
        ax.plot(   t_exac_scaled_tensor.detach().cpu().numpy(), u_exac_scaled_tensor[:, i].detach().cpu().numpy(), label="Exact", color="blue"   , alpha=1)
        ax.scatter(t_data_scaled_tensor.detach().cpu().numpy(), u_data_scaled_tensor[:, i].detach().cpu().numpy(), label="Data" , color="black" , alpha=1, s=30)
        ax.plot(   t_test_scaled_tensor.detach().cpu().numpy(), u_test_scaled_tensor[:, i].detach().cpu().numpy(), label="Test" , color="red" , alpha=1, linestyle="--")
        ax.scatter(t_ini_scaled_tensor.detach().cpu() , u_ini_scaled_tensor[:,i].detach().cpu()  , s=50, label= "Initial Scaled", color = "brown", alpha = 1)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(labels[i])
        ax.grid()
        ax.legend()
    plt.show()
    
    
    
    
    #%% 6. Train Network with exact data
    model = PINNs_MLPs([1,128,128,2]).to(device)  # ورودی t، خروجی (H, α)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_weights_tensor = torch.tensor([0,0,0,0,0,0,1,1], dtype = torch.float32, device = device)#ok - 30% 100000
    #loss_weights_tensor = torch.tensor([ 1, 1, 0.1, 0.1, .01, .01, 15, 15], dtype = torch.float32, device = device)
    #loss_weights_tensor = torch.tensor([ 1, 1, 0.1, 0.1, .01, .01, 20, 20], dtype = torch.float32, device = device)#should run
    #loss_weights_tensor = torch.tensor([ 1, 1, 0.1, 0.1, .01, .01, 40, 40], dtype = torch.float32, device = device)#should run
    loss_weights_tensor = torch.tensor([ 1, 1, 0.1, 0.1, .01, .01, 50, 50], dtype = torch.float32, device = device)#should run
    loss_weights_tensor = torch.tensor([ .1, .1, 0.1, 0.1, .005, .05, 150, 150], dtype = torch.float32, device = device)#should run tend=6 lr = 0.01 ok
    loss_weights_tensor = torch.tensor([0.015, 0.015, 1500.0, 1.0, 10.0, 5.0, 1.0, 1.0], dtype = torch.float32, device = device)#should run tend=10
    loss_weights_tensor = torch.tensor([0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0], dtype = torch.float32, device = device)#should run tend=10 lr = 0.01 ok
    loss_weights_tensor = torch.tensor([0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0], dtype = torch.float32, device = device)#should run tend=6 lr = 0.001 ok data nata=20
    loss_weights_tensor = torch.tensor([0.015, 0.015, 10, 10, 10.0, 10.0, 50.0, 50.0], dtype = torch.float32, device = device)#should run tend=6 lr = 0.001 ok data nata=15


    
    ODE_indices = [0,1]
    IC_indices = [2,3,4,5]
    Data_indices = [6,7]
    
    #loss_weights_tensor[ODE_indices] /= loss_weights_tensor[ODE_indices].sum()
    #loss_weights_tensor[IC_indices] /= loss_weights_tensor[IC_indices].sum()
    #loss_weights_tensor[Data_indices] /= loss_weights_tensor[Data_indices].sum()
    loss_weights_tensor /= loss_weights_tensor.sum()
    
    print("W:",loss_weights_tensor.detach().cpu().numpy())
    
    for i in range(30000):
        optimiser.zero_grad(set_to_none=True)
        current_lr      = optimiser.param_groups[0]["lr"]
        
        data     = data_loss(model, t_data_scaled_tensor, u_data_scaled_tensor)
        #print(t_data_scaled_tensor.shape,u_data_scaled_tensor.shape)  # should check
        ode      = ode_loss(model, t_coll_scaled_tensor,M_bar_torch, K_bar_torch, K_0_torch, k3_torch, e1_torch, e2_torch, sfh_tensor,smh_tensor, sfa_tensor,sma_tensor)
        ic       = ic_loss(model, t_ini_scaled_tensor,z0_scaled_tensor)
    
    
        loss_raw        = torch.stack([ode[0], ode[1], ic[0], ic[1], ic[2], ic[3], data[0], data[1]])
        loss_weighted   = loss_weights_tensor * loss_raw
        loss_total      = torch.sum(loss_weighted)
        l2er =L2_relative_error(model, t_test_scaled_tensor, u_test_scaled_tensor)
        loss_total.backward()
        optimiser.step()
        #scheduler.step()
        if i % 1000 == 0:
            print(
                f"Epoch {i:5d}, Loss: {loss_total.item():.4f}, LR: {current_lr:.2e}, "
                f"L2E: {l2er.item():.4f}, RLoss: [{', '.join(f'{x:.4e}' for x in loss_raw.detach().cpu().numpy())}]"
            )
    # Sure Network with much Data Work
    print("\n--- Starting LBFGS Optimization ---\n")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=50000,
        max_eval=50000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe"  # استفاده از line search قوی‌تر
    )
    
    def closure():
        optimizer_lbfgs.zero_grad(set_to_none=True)
    
        data = data_loss(model, t_data_scaled_tensor, u_data_scaled_tensor)
        ode      = ode_loss(model, t_coll_scaled_tensor,M_bar_torch, K_bar_torch, K_0_torch, k3_torch,
                            e1_torch, e2_torch, sfh_tensor,smh_tensor, sfa_tensor,sma_tensor)

        ic   = ic_loss(model, t_ini_scaled_tensor, z0_scaled_tensor)
    
        loss_raw = torch.stack([ode[0],ode[1], ic[0],ic[1], ic[2],ic[3], data[0],data[1]])
        loss_weighted   = loss_weights_tensor * loss_raw
        loss_total      = torch.sum(loss_weighted)
    
        loss_total.backward()
        return loss_total
    
    # اجرای LBFGS
    lbfgs_iter = 0
    max_lbfgs_iter = 500
    
    while lbfgs_iter < max_lbfgs_iter:
        loss_value = optimizer_lbfgs.step(closure)
        lbfgs_iter += 1
    
        if lbfgs_iter % 50 == 0:
            with torch.no_grad():
                l2er = L2_relative_error(model, t_test_scaled_tensor, u_test_scaled_tensor)
                print(f"[LBFGS] Iter {lbfgs_iter:4d}, Loss: {loss_value.item():.4e}, L2E: {l2er.item():.4e}")
    
    print("\nTraining completed.")
    # Sure Network with much Data Work
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
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #plt.savefig(f"figures/plot_{timestamp}_.png")
    
    #print(best_params)
    print(loss_raw)
    print(loss_weighted)
    
    

    save_globals("mass_spring_pinns_train.pth", scope=globals())
    torch.save(model.state_dict(), "mass_spring_pinns_network.pth")

    
    
    

    #os.system("shutdown /h")