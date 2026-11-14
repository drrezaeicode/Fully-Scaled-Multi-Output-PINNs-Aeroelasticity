# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 00:23:50 2025

@author: Mohsen
"""

# -*- coding: utf-8 -*-
"""
save_load_function.py
"""

import torch
import numpy as np


def _is_numeric_seq(x):
    """لیست/تاپل غیرخالی که فقط عدد (نه bool) دارد، مثل loss_history."""
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            for v in x
        )
    )


def save_globals(filename="globals.pth", scope=None):
    """
    scope = globals() اسکریپت اصلی که می‌خواهی ذخیره شود.
    ذخیره می‌شوند: تنسورها، آرایه‌های numpy، و لیست/تاپل‌های عددی (به صورت np.ndarray).
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
        elif _is_numeric_seq(obj):
            workspace[name] = np.asarray(obj)

    torch.save(workspace, filename)
    print(f"✅ {len(workspace)} items saved to {filename}")


def load_globals(filename="globals.pth", scope=None):
    """
    بارگذاری آیتم‌های ذخیره‌شده به globals.
    scope = globals() اسکریپت اصلی که می‌خواهی متغیرها به آن اضافه شود.
    """
    if scope is None:
        raise ValueError("You must pass the globals() of your script")

    # weights_only=False برای سازگاری با PyTorch 2.6 و بالاتر
    loaded_workspace = torch.load(filename, map_location="cpu", weights_only=False)
    scope.update(loaded_workspace)

    print(f"✅ {len(loaded_workspace)} items loaded from {filename}")
    print("---- Loaded variables ----")
    for name, obj in loaded_workspace.items():
        print(name, type(obj), getattr(obj, "shape", ""))

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
    
    # اضافه کردن weights_only=False برای سازگاری با PyTorch 2.6 و بالاتر
    loaded_workspace = torch.load(filename, map_location="cpu", weights_only=False)
    scope.update(loaded_workspace)
    
    print(f"✅ {len(loaded_workspace)} tensors/arrays loaded from {filename}")
    print("---- Loaded variables ----")
    for name, obj in loaded_workspace.items():
        shp = getattr(obj, 'shape', '')
        print(name, type(obj), shp)
    
    return loaded_workspace



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