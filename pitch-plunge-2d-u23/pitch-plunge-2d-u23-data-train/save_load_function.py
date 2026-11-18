# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 00:23:50 2025

@author: Mohsen
"""


import torch
import numpy as np

    
def save_globals(filename="globals.pth", scope=None):
    """
    scope = globals() اسکریپت اصلی که میخوای ذخیره بشه
    """
    if scope is None:
        raise ValueError("You must pass the globals() of your script")
    
    workspace = {}
    for name, obj in list(scope.items()):
        if name.startswith("__"):
            continue
        if isinstance(obj, torch.Tensor):
            workspace[name] = obj.detach().cpu()
        elif isinstance(obj, np.ndarray):
            workspace[name] = obj.copy()
    
    torch.save(workspace, filename)
    print(f"✅ {len(workspace)} tensors/arrays saved to {filename}")




def load_globals(filename="globals.pth", scope=None):
    """
    بارگذاری تانسورها و آرایه‌های ذخیره شده به globals
    scope = globals() اسکریپت اصلی که میخوای متغیرها اضافه بشه
    """
    if scope is None:
        raise ValueError("You must pass the globals() of your script")
    
    # اضافه کردن weights_only=False برای سازگاری با PyTorch 2.6 و بالاتر
    loaded_workspace = torch.load(filename, map_location="cpu", weights_only=False)
    scope.update(loaded_workspace)
    
    print(f"✅ {len(loaded_workspace)} tensors/arrays loaded from {filename}")
    print("---- Loaded variables ----")
    for name, obj in loaded_workspace.items():
        shp = getattr(obj, 'shape', '')
        print(name, type(obj), shp)
    
    return loaded_workspace


'''
import torch
import numpy as np

    
def save_globals(filename="globals.pth", scope=None):
    """
    scope = globals() اسکریپت اصلی که میخوای ذخیره بشه
    """
    if scope is None:
        raise ValueError("You must pass the globals() of your script")
    
    workspace = {}
    for name, obj in list(scope.items()):
        if name.startswith("__"):
            continue
        if isinstance(obj, torch.Tensor):
            workspace[name] = obj.detach().cpu()
        elif isinstance(obj, np.ndarray):
            workspace[name] = obj.copy()
    
    torch.save(workspace, filename)
    print(f"✅ {len(workspace)} tensors/arrays saved to {filename}")




def load_globals(filename="globals.pth", scope=None):
    """
    بارگذاری تانسورها و آرایه‌های ذخیره شده به globals
    scope = globals() اسکریپت اصلی که میخوای متغیرها اضافه بشه
    """
    if scope is None:
        raise ValueError("You must pass the globals() of your script")
    
    loaded_workspace = torch.load(filename, map_location="cpu")
    scope.update(loaded_workspace)
    
    print(f"✅ {len(loaded_workspace)} tensors/arrays loaded from {filename}")
    print("---- Loaded variables ----")
    for name, obj in loaded_workspace.items():
        shp = getattr(obj, 'shape', '')
        print(name, type(obj), shp)
    
    return loaded_workspace
'''