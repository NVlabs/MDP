"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import torch
import torch.nn as nn
from apex import amp
import pdb

'''
Gating layer for pruning
'''


class GateLayer(nn.Module):
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))

        # for simpler way to find these layers
        self.do_not_update = True

    # @amp.half_function
    def forward(self, input):
        return input*self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )


class GateLayerGumbel(nn.Module):
    def __init__(self, output_features, temperature=0.3, bch_size=64, dim=1):
        super(GateLayerGumbel, self).__init__()
        self.output_features = output_features
        self.do_not_update = True

        prob_init = 0.05 * torch.ones(output_features)  + torch.randn(output_features)*0.05
        prob_init = torch.randn(output_features)*0.01
        prob_init = torch.clamp(prob_init, 0.01, 0.99)

        self.logits = False
        self.logits = True
        eps = 1e-5
        if self.logits:
            prob_init = torch.log(prob_init / (1 - prob_init + eps) + eps)

        self.weight = nn.Parameter(prob_init)
        self.temperature = temperature

        #if disrete then will sample only once
        self.discrete = False

        self.bch_size = bch_size
        self.dim=dim

    def sample(self):
        bch_size = self.bch_size
        if self.discrete:
            nfeatures = self.output_features

            # if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
            #     print(self.weight)

            probabilities = 1.0 - torch.sigmoid(self.weight)
            threshold = 0.5
            probabilities.data[probabilities.data < threshold] = 0.0
            probabilities.data[probabilities.data >= threshold] = 1.0

            mask = probabilities.expand((bch_size, nfeatures))

            output = input * mask.view((bch_size, self.output_features, 1, 1))
            return output

        if self.logits:
            probabilities = torch.sigmoid(self.weight)
        else:
            clamp_threshold = 0.001
            p_clamp = torch.clamp(self.weight, clamp_threshold, 1.0 - clamp_threshold)
            probabilities = p_clamp + 2 * clamp_threshold * (torch.sigmoid(self.weight - p_clamp) - 0.5)

        eps = 1e-8
        if self.logits:
            eps = 1e-10
            eps = 1e-5

        # need this check otherwise have nan
        if torch.isnan(self.weight.sum()):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
                print("fix gate with nan")
                pdb.set_trace()
            self.weight[torch.isnan(self.weight)].data = self.weight[torch.isnan(self.weight)].data*0.0+0.0
            probabilities = torch.sigmoid(self.weight)


        nfeatures = self.output_features

        probabilities = probabilities.view((1, -1)).expand((bch_size, nfeatures))

        prob2 = torch.log(probabilities + eps)#.half()
        prob1 = torch.log(1.0 - probabilities + eps)#.half()

        double_weight = torch.cat((prob2.view((-1, 1)), prob1.view((-1, 1))), 1)

        temperature = 0.01
        # for inference temperature is fixed and is very small
        if self.training:
            # print("setting temperature?")
            temperature = self.temperature

        hard_distribution = False

        mask = my_gumbel_softmax(double_weight, tau=temperature, hard=hard_distribution, eps=1e-5)

        while torch.isnan(mask.sum()):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
                print("nan in gumbel1")
                pdb.set_trace()

        mask = mask.view((bch_size, nfeatures, 2))

        self.mask = mask[:, :, 1]


    def forward(self, input):
        bch_size = self.bch_size

        # pdb.set_trace()
        if self.dim == 1:
            if len(input.shape)==1:
                output = input * self.mask.view((bch_size, self.output_features, 1, 1))
            elif len(input.shape)==3:
                output = input * self.mask.view((bch_size, self.output_features, 1))
        elif self.dim == 2:
            if len(input.shape)==3:
                output = input * self.mask.view((bch_size, 1, self.output_features))

        return output

    def extra_repr(self):
        return ' out_features={}'.format(
             self.output_features
        )


def my_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """


    gumbels = -(torch.empty_like(logits).exponential_()+eps).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret