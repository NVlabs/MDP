import os
import collections
from itertools import repeat
import argparse
import pickle as pkl

import torch.nn as nn
import numpy as np
import torch
import math

from einops import rearrange

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple






class QK(nn.Module):
    def __init__(self, emb, qk, head, qkv_bias=True):
        super().__init__()
        self.head = head
        self.qk_dim = qk * head
        self.Q = nn.Linear(emb, self.qk_dim, bias=qkv_bias)
        self.K = nn.Linear(emb, self.qk_dim, bias=qkv_bias)
        self.scale = head ** -0.5
        
    def forward(self, x):
        B, N, C = x.shape
        q = self.Q(x)
        k = self.K(x)
        qk_dim = q.shape[2]
        q = q.reshape(B, N, self.head, qk_dim // self.head).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.head, qk_dim // self.head).permute(0, 2, 1, 3)
        attn_r = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_r.softmax(dim=-1)
        return attn
    


class V_AND_PROJ(nn.Module):
    def __init__(self, emb, v, head, qkv_bias=True):
        super().__init__()
        self.v_dim = v * head
        self.V = nn.Linear(emb, self.v_dim, bias=qkv_bias)
        self.proj = nn.Linear(self.v_dim, emb)
        self.head = head
        
    def forward(self, x, attn):
        B, N, C = x.shape
        v = self.V(x)
        v = v.reshape(B, N, self.head, self.v_dim // self.head).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(-1, 198, self.v_dim)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
# total, group size, # groups

# all_variable_specs = {
#     "EMB": [768, 16, 768//16+1],
#     "HEAD": [12, 1, 12//1+1],
#     "QK": [64, 2, 64//2+1],
#     "V": [64, 2, 64//2+1],
#     "MLP": [3072, 32, 3072//32+1],
# }
all_variable_specs = {
    "EMB": [768, 16, 768//16+1],
    "HEAD": [12, 2, 12//2+1],
    "QK": [64, 8, 64//8+1],
    "V": [64, 8, 64//8+1],
    "MLP": [3072, 16, 3072//16+1],
}
emb_idx_range = np.arange(1, all_variable_specs["EMB"][2])
head_idx_range = np.arange(1, all_variable_specs["HEAD"][2])
qk_idx_range = np.arange(1, all_variable_specs["QK"][2])
v_idx_range = np.arange(1, all_variable_specs["V"][2])
mlp_idx_range = np.arange(1, all_variable_specs["MLP"][2])

# BS = 256
BS = 576
NUM_TOKENS = 198

WARMUP = 20
TOTAL = 40

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)


def estimate_latency(model, dummy_inputs):
    times = []
    with torch.no_grad():
        for i in range(TOTAL):
            start_evt.record()
            _ = model(*dummy_inputs)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_time = start_evt.elapsed_time(end_evt)
            if i < WARMUP:
                continue
            times.append(elapsed_time)
    
    return np.mean(times)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Latency Estimation Tools.')

    # Add a string argument
    parser.add_argument(
        '--lut-name',  # Name of the argument
        type=str,        # Type of the argument
        help='which lut to profile.'  # Description of the argument
    )

    args = parser.parse_args()
    lut_name = args.lut_name
    assert lut_name in ["qk", "vandproj", "mlp"]
    
    # save_name = f"{lut_name}_lut_BS{BS}_NUM_TOKENS{NUM_TOKENS}_v100.pkl"
    save_name = f"new_{lut_name}_lut_BS{BS}_NUM_TOKENS{NUM_TOKENS}_v100.pkl"
    if os.path.isfile(save_name):
        print("file exists", save_name)
        save_name = "copy" + save_name
    # head, emb, qk
    if lut_name == "qk":
        lut = np.zeros((all_variable_specs["HEAD"][2], all_variable_specs["EMB"][2], all_variable_specs["QK"][2]))
        for head_idx in head_idx_range:
            head = head_idx * all_variable_specs["HEAD"][1]
            print(f"progress {head}/12")
            for emb_idx in emb_idx_range:
                for qk_idx in qk_idx_range:
                    emb = emb_idx * all_variable_specs["EMB"][1]
                    qk = qk_idx * all_variable_specs["QK"][1]
                    # data
                    dummy_QK_input = torch.zeros((BS, NUM_TOKENS, emb))
                    dummy_QK_input = dummy_QK_input.cuda()
                    # model
                    QK_model = QK(emb=emb, qk=qk, head=head)
                    QK_model = QK_model.cuda()
                    QK_model.eval()
                    QK_model = torch.compile(QK_model, backend="tensorrt")
                    QK_model(dummy_QK_input)
                    print("TRT conversion successful")
                    latency = estimate_latency(QK_model, [dummy_QK_input])
                    lut[head_idx, emb_idx, qk_idx] = latency
        
        with open(save_name, 'wb') as f:
            pkl.dump(lut, f)
        return
    
    # head, emb, v
    elif lut_name == "vandproj":
        lut = np.zeros((all_variable_specs["HEAD"][2], all_variable_specs["EMB"][2], all_variable_specs["V"][2]))
        for head_idx in head_idx_range:
            head = head_idx * all_variable_specs["HEAD"][1]
            print(f"progress {head}/12")
            for emb_idx in emb_idx_range:
                for v_idx in v_idx_range:
                    emb = emb_idx * all_variable_specs["EMB"][1]
                    v = v_idx * all_variable_specs["V"][1]
                    # data
                    dummy_V_input = torch.zeros((BS, NUM_TOKENS, emb))
                    dummy_V_input = dummy_V_input.cuda()
                    dummy_attn = torch.zeros((BS, head, NUM_TOKENS, NUM_TOKENS))
                    dummy_attn = dummy_attn.cuda()
                    # model
                    V_AND_PROJ_MODEL = V_AND_PROJ(emb=emb, v=v, head=head)
                    V_AND_PROJ_MODEL = V_AND_PROJ_MODEL.cuda()
                    V_AND_PROJ_MODEL.eval()
                    V_AND_PROJ_MODEL = torch.compile(V_AND_PROJ_MODEL, backend="tensorrt")
                    V_AND_PROJ_MODEL(dummy_V_input, dummy_attn)
                    print("TRT conversion successful")
                    latency = estimate_latency(V_AND_PROJ_MODEL, [dummy_V_input, dummy_attn])
                    lut[head_idx, emb_idx, v_idx] = latency
        
        with open(save_name, 'wb') as f:
            pkl.dump(lut, f)
        return
    
    # emb, mlp
    elif lut_name == "mlp":
        lut = np.zeros((all_variable_specs["EMB"][2], all_variable_specs["MLP"][2]))
        for emb_idx in emb_idx_range:
            print(f"progress {emb_idx}/{emb_idx_range[-1]}")
            for mlp_idx in mlp_idx_range:
                emb = emb_idx * all_variable_specs["EMB"][1]
                mlp = mlp_idx * all_variable_specs["MLP"][1]
                # data
                dummy_mlp_input = torch.zeros((BS, NUM_TOKENS, emb))
                dummy_mlp_input = dummy_mlp_input.cuda()
                # model
                MLP_MODEL = Mlp(emb, mlp)
                MLP_MODEL = MLP_MODEL.cuda()
                MLP_MODEL.eval()
                MLP_MODEL = torch.compile(MLP_MODEL, backend="tensorrt")
                MLP_MODEL(dummy_mlp_input)
                print("TRT conversion successful")
                latency = estimate_latency(MLP_MODEL, [dummy_mlp_input])
                lut[emb_idx, mlp_idx] = latency
        
        with open(save_name, 'wb') as f:
            pkl.dump(lut, f)
        return
    
if __name__ == '__main__':
    main() 