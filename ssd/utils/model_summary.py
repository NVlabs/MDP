import torch
import pandas as pd
from functools import partial
import numpy as np


def volume(tensor):
    """return the volume of a pytorch tensor"""
    if isinstance(tensor, torch.FloatTensor) or isinstance(
        tensor, torch.cuda.FloatTensor
    ):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert (
        isinstance(torch_size, torch.Size)
        or isinstance(torch_size, tuple)
        or isinstance(torch_size, list)
    )
    return "(" + (", ").join(["%d" % v for v in torch_size]) + ")"


def model_find_module_name(model, module_to_find):
    """Look up the name of a module in a model.
    Arguments:
        model: the model to search
        module_to_find: the module whose name we want to look up
    Returns:
        The module name (string) or None, if the module was not found.
    """
    for name, m in model.named_modules():
        if m == module_to_find:
            return name
    return None


def conv_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Conv2d)
    if self in memo:
        return

    weights_vol = (
        self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]
    )

    # Multiply-accumulate operations: MACs = volume(OFM) * (#IFM * K^2) / #Groups
    # Bias is ignored
    macs = volume(output) * (
        self.in_channels / self.groups * self.kernel_size[0] * self.kernel_size[1]
    )
    attrs = "k=" + "(" + (", ").join(["%d" % v for v in self.kernel_size]) + ")"
    module_visitor(self, input, output, df, model, weights_vol, macs, attrs)


def module_visitor(self, input, output, df, model, weights_vol, macs, attrs=None):
    in_features_shape = input[0].size()
    out_features_shape = output.size()

    mod_name = model_find_module_name(model, self)
    df.loc[len(df.index)] = [
        mod_name,
        self.__class__.__name__,
        attrs if attrs is not None else "",
        size_to_str(in_features_shape),
        volume(input[0]),
        size_to_str(out_features_shape),
        volume(output),
        int(weights_vol),
        int(macs),
    ]


def fc_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Linear)
    if self in memo:
        return

    # Multiply-accumulate operations: MACs = #IFM * #OFM
    # Bias is ignored
    weights_vol = macs = self.in_features * self.out_features
    module_visitor(self, input, output, df, model, weights_vol, macs)


def model_performance_summary(model, dummy_input, batch_size=1):
    """Collect performance data"""

    def install_perf_collector(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(
                m.register_forward_hook(
                    partial(conv_visitor, df=df, model=model, memo=memo)
                )
            )
        elif isinstance(m, torch.nn.Linear):
            hook_handles.append(
                m.register_forward_hook(
                    partial(fc_visitor, df=df, model=model, memo=memo)
                )
            )

    df = pd.DataFrame(
        columns=[
            "Name",
            "Type",
            "Attrs",
            "IFM",
            "IFM volume",
            "OFM",
            "OFM volume",
            "Weights volume",
            "MACs",
        ]
    )

    hook_handles = []
    memo = []

    model.apply(install_perf_collector)
    # Now run the forward path and collect the data
    model(dummy_input)
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    return df


def performance_summary(model, dummy_input, opt=None, prefix=""):
    try:
        df = model_performance_summary(model.module, dummy_input)
    except AttributeError:
        df = model_performance_summary(model, dummy_input)
    new_entry = {
        "Name": ["Total"],
        "MACs": [df["MACs"].sum()],
    }
    MAC_total = df["MACs"].sum()
    return MAC_total


def model_summary(model, dummy_input, opt=None):
    return (
        performance_summary(model, dummy_input, opt)
    )