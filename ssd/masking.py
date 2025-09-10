# For Soft Masking
# Add forward prehooks to the model

from enum import Enum
from typing import Any, Callable, List, Tuple

import torch
from torch import autograd, nn

class ParameterMaskingType(Enum):
    Soft = "Soft"
    Hard = "Hard"
    Permanent = "Permanent"

class SoftParameterMasker(autograd.Function):
    """Dynamic STE (straight-through estimator) parameter masker"""

    @staticmethod
    def forward(_, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weight * mask

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None
    
class HardParameterMasker(autograd.Function):
    """Hard parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        mask, = ctx.saved_tensors
        return grad_output * mask, None


def get_orig(module: nn.Module, name: str) -> nn.Parameter:
    if has_mask(module, name):
        return getattr(module, f"{name}_orig")
    else:
        return getattr(module, name)

def has_mask(module: nn.Module, name: str) -> bool:
    return hasattr(module, f"{name}_mask")

def get_mask(module: nn.Module, name: str) -> torch.Tensor:
    return getattr(module, f"{name}_mask")

def set_mask(module: nn.Module, name: str, new_mask: torch.Tensor) -> None:
    """
    Can pass any mask that is expandable to the actual mask size
    """
    if not has_mask(module, name):
        assert new_mask.all(), "Cannot set a non-True mask to an unmasked layer"
        return

    masking_type: ParameterMaskingType = getattr(module, f"{name}_masking_type")

    if masking_type == ParameterMaskingType.Permanent:
        # If masking is permanent, immediately apply mask to orig and set mask zeros
        orig = get_orig(module, name)
        orig.data[~new_mask.expand(*orig.shape)] = 0

        mask = get_mask(module, name)
        mask.data[~new_mask.expand(*mask.shape)] = 0
    else:
        # Otherwise set the mask as usual
        mask = get_mask(module, name)
        mask.data[:] = new_mask.expand(*mask.shape)
        

def _create_masking_pre_hook(name: str, masking_type: ParameterMaskingType) -> Callable[[nn.Module, Any], None]:
    if masking_type == ParameterMaskingType.Soft:
        masker = SoftParameterMasker
    elif masking_type == ParameterMaskingType.Hard or masking_type == ParameterMaskingType.Permanent:
        masker = HardParameterMasker
    else:
        raise NotImplementedError(f"Forward pre hook for masking type '{masking_type}' not implemented")

    def _masking_pre_hook(module: nn.Module, _: Any) -> None:
        assert hasattr(module, f"{name}_orig"), f"Module {module} param/buffer {name} wasn't setup to be pruned"

        mask = get_mask(module, name)
        orig = get_orig(module, name)

        setattr(module, name, masker.apply(orig, mask))

    return _masking_pre_hook

def _create_masking_cleanup_hook(name: str) -> Callable[[nn.Module, Any], None]:
    def _masking_cleanup_hook(module: nn.Module, input: Any, output: Any) -> None:
        assert hasattr(module, f"{name}_orig"), f"Module {module} param/buffer {name} wasn't setup to be pruned"
        # print(module.weight)
        setattr(module, name, None)

    return _masking_cleanup_hook

def add_attr_masking(
    module: nn.Module, name: str, masking_type: ParameterMaskingType, mask_zeros: bool
) -> None:
    if not hasattr(module, name):
        return
    orig = getattr(module, name)
    if orig is None or name not in module._parameters.keys():
        return

    # Move original parameter to new name
    module.register_parameter(f"{name}_orig", orig)
    del module._parameters[name]

    # Store the masking type
    setattr(module, f"{name}_masking_type", masking_type)

    # Register the masking buffer
    mask = (orig != 0.) if mask_zeros else torch.ones_like(orig, dtype=torch.bool)
    module.register_buffer(f"{name}_mask", mask)

    # Register forward pre hook that will recompute the plain attribute before each forward
    remove_prehook = module.register_forward_pre_hook(_create_masking_pre_hook(name, masking_type))
    setattr(module, f"{name}_remove_prehook", remove_prehook)

    # Register forward hook that will cleanup the plain attribute after each forward
    # Note: without this, the model cannot be deepcopied (due to non-leaf tensors)
    remove_posthook = module.register_forward_hook(_create_masking_cleanup_hook(name))
    setattr(module, f"{name}_remove_posthook", remove_posthook)

def remove_attr_masking(module: nn.Module, name: str) -> None:
    orig = module._parameters.get(f"{name}_orig", None)

    if orig is None:
        return

    # update the orig with the current mask
    mask = module._buffers[f"{name}_mask"]
    orig.data = (orig * mask).detach()

    # Remove the masking type
    delattr(module, f"{name}_masking_type")

    # delete
    remove_prehook = getattr(module, f"{name}_remove_prehook")
    remove_prehook.remove()
    delattr(module, f"{name}_remove_prehook")

    remove_posthook = getattr(module, f"{name}_remove_posthook")
    remove_posthook.remove()
    delattr(module, f"{name}_remove_posthook")

    del module._buffers[f"{name}_mask"]
    del module._parameters[f"{name}_orig"]

    setattr(module, name, orig)
        
