"""
Hook management for capturing model activations.
"""

import torch
from config import LAYER_NAME


class HookManager:
    def __init__(self, model, layer_name=LAYER_NAME):
        self.model = model
        self.layer_name = layer_name
        self.target_layer = dict()
        self.hook_handle = None
        
    def capture_activations(self, module, inp, out):
        """Hook function to capture activations."""
        out = out if isinstance(out, torch.Tensor) else out[0]
        if not out.requires_grad:
            out.requires_grad_()
        out.retain_grad()
        self.target_layer['output'] = out
        print(f"[DEBUG] Hooked layer output shape: "
              f"{out.shape if isinstance(out, torch.Tensor) else type(out)}")
    
    def register_hook(self):
        """Register the forward hook on the specified layer."""
        hooked_layer = eval(f"self.model.{self.layer_name}")
        self.hook_handle = hooked_layer.register_forward_hook(self.capture_activations)
        print(f"Hook registered on layer: {self.layer_name}")
        return self.hook_handle
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            print("Hook removed successfully")
        else:
            print("No hook to remove")
    
    def get_captured_output(self):
        """Get the captured activation output."""
        return self.target_layer.get('output', None)
