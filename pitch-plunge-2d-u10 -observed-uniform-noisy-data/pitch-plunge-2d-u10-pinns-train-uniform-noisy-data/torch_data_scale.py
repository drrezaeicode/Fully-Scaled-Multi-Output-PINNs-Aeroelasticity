# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:50:16 2025

@author: Mohsen
"""


#import torch


class MinMaxScalerPytorchTX:
    def __init__(self, t_range = (-1, 1),x_range = (-1, 1),device = 'cpu'):
        self.device = device
        
        # Feature ranges
        self.ts_min, self.ts_max = t_range  # Min and Max time scaled value
        self.xs_min, self.xs_max = x_range   # Min and Max X scaled value        
        
        # for t
        self.tr_min = None   # Min time real value, time tensor size (N,1)
        self.tr_max = None   # Max time real value 
        self.tscale_ = None
        self.tmin_ = None        
        
        
        self.xr_min = None   # Min X real value, X tensor size (N,K)
        self.xr_max = None   # Max X real value
        self.xscale_ = None
        self.xmin_ = None
        
    def fit(self, t, x):
        t = t.to(self.device)
        x = x.to(self.device)
        
        self.tr_min = t.min(dim=0, keepdim=True).values
        self.tr_max = t.max(dim=0, keepdim=True).values
        self.tscale_ = (self.ts_max - self.ts_min)/(self.tr_max - self.tr_min)
        self.tmin_   = self.ts_min - self.tscale_ * self.tr_min        
        
        self.xr_min = x.min(dim=0, keepdim=True).values
        self.xr_max = x.max(dim=0, keepdim=True).values        
        self.xscale_ = (self.xs_max - self.xs_min)/(self.xr_max - self.xr_min)
        self.xmin_   = self.xs_min - self.xscale_ * self.xr_min
        return self
    
    def transform(self, t, x):
        t = t.to(self.device)
        x = x.to(self.device)
        t_scaled = t * self.tscale_ + self.tmin_
        x_scaled = x * self.xscale_ + self.xmin_
        return t_scaled, x_scaled
    
    def transformT(self,t):
        t = t.to(self.device)
        t_scaled = t * self.tscale_ + self.tmin_
        return t_scaled
    
    def transformX(self,t):
        x = t.to(self.device)
        x_scaled = x * self.xscale_ + self.xmin_
        return x_scaled
    
    def inverse_transform(self, t_scaled, x_scaled):
        t_scaled = t_scaled.to(self.device)
        x_scaled = x_scaled.to(self.device)
        t_orgin = (t_scaled - self.tmin_) / self.tscale_
        x_orgin = (x_scaled - self.xmin_) / self.xscale_
        return t_orgin, x_orgin
    
    def inverse_transformT(self, t_scaled):
        t_scaled = t_scaled.to(self.device)
        t_orgin = (t_scaled - self.tmin_) / self.tscale_
        return t_orgin
    
    def inverse_transformX(self, x_scaled):
        x_scaled = x_scaled.to(self.device)
        x_orgin = (x_scaled - self.xmin_) / self.xscale_
        return x_orgin
    
    def fit_transform(self, T,X):
        return self.fit(T,X).transform(T,X)
    
    def scale_first_derivative_transform(self, dx_dt):
        dx_dt = dx_dt.to(self.device)
        return dx_dt * (self.xscale_ / self.tscale_)
    
    def scale_second_derivative_transform(self, d2x_dt2):
       d2x_dt2 = d2x_dt2.to(self.device)
       return d2x_dt2 * (self.xscale_ / (self.tscale_**2))
    
    def scale_first_derivative_inverse_transform(self, dx_dt_scaled):
        dx_dt_scaled = dx_dt_scaled.to(self.device)
        return dx_dt_scaled * (self.tscale_ / self.xscale_)
    
    def scale_second_derivative_inverse_transform(self, d2x_dt2_scaled):
        d2x_dt2_scaled = d2x_dt2_scaled.to(self.device)
        return d2x_dt2_scaled * ((self.tscale_**2) / self.xscale_)    
   
'''
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Using Device:{device}')
    
T = torch.tensor([[3],[5],[7],[9]]).to(device)
X = torch.tensor([[1,2],[3,4],[5,6],[7,8]]).to(device)

print(X)      
scaler = MinMaxScalerPytorchTX(t_range = (0, 1),x_range = (-1, 1),device=device).fit(T,X)
T_scaled, X_scaled = scaler.transform(T,X)
print(T_scaled, X_scaled)
print(T_scaled.shape, X_scaled.shape)
print("sclae factor:",scaler.tscale_, scaler.xscale_)
print("minfactor:", scaler.tmin_, scaler.xmin_)
T_original,X_original = scaler.inverse_transform(T_scaled,X_scaled)
print(T_original, X_original)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

# ---------------------------
# Example Data
# ---------------------------
T = torch.tensor([[0.],[1.],[2.],[3.],[4.]], dtype=torch.float32, device=device)   # Time vector
X = torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]], dtype=torch.float32, device=device)  # Feature matrix

# Example derivatives in original space
dx_dt = torch.tensor([[1,1],[1,1],[1,1],[1,1],[1,1]], dtype=torch.float32, device=device)
d2x_dt2 = torch.tensor([[0.1,0.1],[0.1,0.1],[0.1,0.1],[0.1,0.1],[0.1,0.1]], dtype=torch.float32, device=device)

# ---------------------------
# Initialize and Fit Scaler
# ---------------------------
scaler = MinMaxScalerPytorchTX(t_range=(0,1), x_range=(-1,1), device=device).fit(T,X)

# Transform T and X
T_scaled, X_scaled = scaler.transform(T,X)
print("T_scaled:\n", T_scaled)
print("X_scaled:\n", X_scaled)

# Inverse transform to recover original
T_orig, X_orig = scaler.inverse_transform(T_scaled, X_scaled)
print("T_original:\n", T_orig)
print("X_original:\n", X_orig)

# ---------------------------
# Scale derivatives
# ---------------------------
dx_scaled = scaler.scale_first_derivative(dx_dt)
d2x_scaled = scaler.scale_second_derivative(d2x_dt2)
print("dx_scaled/dt_scaled:\n", dx_scaled)
print("d2x_scaled/dt_scaled^2:\n", d2x_scaled)

# ---------------------------
# Inverse scale derivatives
# ---------------------------
dx_orig = scaler.scale_first_derivative_inverse_transform(dx_scaled)
d2x_orig = scaler.scale_second_derivative_inverse_transform(d2x_scaled)
print("dx original:\n", dx_orig)
print("d2x original:\n", d2x_orig)
'''