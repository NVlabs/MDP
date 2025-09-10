"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# Pruning optimizer for sparse pruning

from torch.optim.optimizer import Optimizer
import torch
import pdb


# 1D m:n structure
def mn_1d(mat, m, n, dynamic_ratio = 1.0):
    # dynamically increase the ratio for columns to be removed
    # Reshape matrix into m-dimensional vectors.
    shape = mat.shape
    mat = mat.view(-1, m)

    # Sort elements by magnitude for each vector.
    sorted_a, idx_a = torch.sort(mat.abs())

    # Set (m-n) elements with lowest magnitude in each vector to zero.
    mask = torch.ones(mat.shape).int().type(mat.type())
    zeros = torch.zeros(mat.shape).int().type(mat.type())

    if dynamic_ratio != 1.0:
        # rows:
        dynamic_process = int(mat.shape[0] * dynamic_ratio)
        mask.scatter_(1, idx_a[:dynamic_process, :(m - n)], zeros)
    else:
        mask.scatter_(1, idx_a[:, :(m - n)], zeros)

    return mask.view(shape).contiguous()


###########################################################################################
# for 2d conv

def blockshaped(arr, m, n):
    """
    Return an array of shape (h/m, w/n, m, n).
    """
    h, w = arr.shape
    return (arr.reshape(h // m, m, -1, n).permute(0, 2, 1, 3).reshape(h // m, w // n, m * n))


def convert_to_mask(valid):
    """
    Converts a list of integers to an array of masks.
    """
    patterns = torch.zeros(len(valid), 4, 4)
    for i, number in enumerate(valid):
        aux_number = number
        for row in range(4):
            for column in range(4):
                patterns[i, row, column] = aux_number & 1
                aux_number = aux_number >> 1
    return patterns


def num_ones(a):
    """
    Counts the number of non-zeroes in a 4-bit vector.
    """
    n = 0
    for i in range(4):
        n += a & 1
        a = a >> 1
    return n


def valid_patterns_m4n2():
    """
    Loops over all possible masks in a 4x4 block and returns an array of masks that are 4:2
    along both dimensions. Each i represents a 4x4 bit-wise mask of 0s and 1s (e.g. 2 bits
    for 16 elements = 2^16). For example, when i=1 the mask is {0001,0000,0000,0000},
    and when i=37 it is {0111,0010,0000,0000}.
    """
    valid = []
    for i in range(1 << 16):
        n = 0
        for row in range(4):
            offset = 4 * row
            mask = (i & (0xf << offset)) >> offset
            n = num_ones(mask)
            if n != 2: break
        if n == 2:
            for column in range(4):
                a = 0
                for row in range(4):
                    offset1 = column + 4 * row
                    offset2 = column + 3 * row
                    a = a | ((i & (1 << offset1)) >> offset2)
                a = a & 0xf
                n = num_ones(a)
                if n != 2: break
            if n == 2: valid.append(i)

    return convert_to_mask(valid)


def best_pattern_sum(mat, pattern_set, m, fraction = 1.0):
    """
    Finds the best mask for a mxm block of weights. Sum of non-masked weights.
    fraction will specify ratio of columns to be transformed, should be 1.0 when pruning finishes
    """
    mask = torch.ones(mat.shape).int()
    mat = blockshaped(mat, m, m)

    # import pdb; pdb.set_trace()
    costs = torch.matmul(mat, pattern_set.view(pattern_set.shape[0], m * m).t().cuda())
    pmax = torch.argmax(costs, dim=2)


    # dynamic to go from low to high criteria
    pmax_costs = torch.max(costs, dim=2)[0]
    dim0 =int(min(fraction, 1.0) * (pmax_costs.numel()-1))
    threshold = pmax_costs.view(-1).sort()[0][dim0]

    sparsity_steps_done = 0

    for row in range(0, mask.shape[0]//m):
        for col in range(0, mask.shape[1]//m):
            if pmax_costs[row, col] <= threshold:
                if sparsity_steps_done >= dim0:
                    break

                mask[row*m:(row+1)*m, col*m:(col+1)*m] = pattern_set[pmax[row, col]]

                sparsity_steps_done += 1

    # for row in range(0, mask.shape[0]//m):
    #     for col in range(0, mask.shape[1]//m):
    #         mask[row*m:(row+1)*m, col*m:(col+1)*m] = pattern_set[pmax[row, col]]

    # original
    # dim0 = int(min(fraction, 1.0) * mask.shape[0])
    # for row in range(0, dim0, m):
    #     for col in range(0, mask.shape[1], m):
    #         mask[row:row + m, col:col + m] = pattern_set[pmax[row // m, col // m]]

    return mask


def mn_2d_best(mat, m, n, fraction):
    """
    Find the best m:n pattern for each mxm block.
    """
    patterns = valid_patterns_m4n2()
    return best_pattern_sum(mat, patterns, m, fraction)


def m4n2_2d_best(mat, density, seed=None, fraction=1.0):
    return mn_2d_best(mat, 4, 2, fraction)


def m4n1_2d_best(mat, density, seed=None):
    return mn_2d_best(mat, 4, 1)


class pruning_optimizer(Optimizer):
    r"""Implements helping functions to count number of FLOPS, number of parameters, remove biases of pruned weights etc.
    """

    def __init__(self, parameters=None, momentum=0.0, method=23,
                 sparsity_pattern="unstructured", dynamic_pruning=False, sparsity_level=2):
        '''
        '''
        #method 23 for Taylor
        #method 2 for weight magnitude

        defaults = dict(momentum = momentum)

        super(pruning_optimizer, self).__init__(parameters, defaults)

        self.pruning_method = method

        self.current_pruning_iteration = 0

        self.sparsity_pattern = sparsity_pattern
        # self.sparsity_pattern = 'unstructured'
        # self.sparsity_pattern = 'ampere'
        # self.sparsity_pattern = 'rows' # to be added soon
        # self.sparsity_pattern = 'cols' # to be added soon

        self.dynamic_pruning = dynamic_pruning
        self.pruning_level = 2
        # if self.dynamic_pruning:
        #     #will go for 25% sparsity and then for 50% sparsity
        #     self.pruning_level = 4
        # else:
        #     self.pruning_level = 2

        self.pruning_level = sparsity_level

        self.iter_num = 0

        self.is_finished = False

        self.full_loss = 0.0
        self.cur_loss = 0.0

        self.use_noise = False

        self.gglobal = False
        self.gglobal = False
        self.thresholds = None

    def __setstate__(self, state):
        super(pruning_optimizer, self).__setstate__(state)

    def step(self, closure=None):
    # def step(self, closure=None, pruning_ratio=0.0, do_pruning=False):
        """doing pruning
        """

        show_once = True

        do_pruning = self.do_pruning
        pruning_ratio = self.pruning_ratio


        loss = None
        if closure is not None:
            loss = closure()

        layer = -1

        if do_pruning:
            #increase clock for how many pruning iterations to do
            self.current_pruning_iteration += 1

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                # if p.grad is None:
                #     continue

                layer += 1

                # if torch.isnan(p.grad.sum()):
                #     # import pdb;
                #     # pdb.set_trace()
                #     print("\n\nPRUNING: skip because grad is NaN\n\n")
                #     break

                param_state = self.state[p]
                if not self.is_finished:
                    weight_size = p.data.numel()

                    ##accumulate scores
                    if 'importance_scores' not in param_state:
                        param_state['importance_scores'] = (p.data.clone()*0.0).type(torch.float)
                        param_state['importance_scores'] = param_state['importance_scores']

                    importance_scores = param_state['importance_scores']
                    #does it work?
                    # print(layer, "\t importance_scores_mean", param_state['importance_scores'].mean())

                    ####### calculate importance scores:
                    if self.sparsity_pattern in ["ampere", "unstructured"]:

                        if self.pruning_method == 23:
                            # Taylor = sqr(weight * gradient)
                            importance_scores.add_( (p.data.type(torch.float) * p.grad.data.type(torch.float)).detach().pow(2) )
                        elif self.pruning_method == 24:
                            # Taylor = abs(weight * gradient)
                            importance_scores.add_(
                                (p.data.clone().type(torch.float) * p.grad.data.clone().type(torch.float)).abs())
                            # importance_scores.add_(
                            #     (p.grad.data.clone().type(torch.float)))

                        elif self.pruning_method == 25:
                            # Taylor = abs(weight * gradient)
                            wd = p.data*3.0517578125e-05
                            wd = 0.0
                            #works better with WD, but hey we should not use it

                            tmp = (p.data.clone().type(torch.float) * ( wd +  p.grad.clone().type(torch.float)) ).abs()

                            if self.use_noise:
                                loss_change = self.cur_loss - self.full_loss
                                per_weight_loss = abs(loss_change/self.total_zero)
                                tmp += ((1.0 - param_state['noise'])*per_weight_loss).type(torch.float)

                            tmp = tmp / (1e-6 + torch.norm(tmp)) # normalization helps with very small values that get ignored otherwise

                            importance_scores.add_(tmp)

                        elif self.pruning_method == 26:
                            # tmp = (p.data.clone() * 0.0).type(torch.float)
                            tmp = (p.data.clone().type(torch.float) * (p.grad.clone().type(torch.float))).abs()

                            if self.use_noise:
                                loss_change = self.cur_loss - self.full_loss
                                per_weight_loss = abs(loss_change/self.total_zero)
                                tmp += ((1.0 - param_state['noise'])*per_weight_loss).type(torch.float)

                            importance_scores.add_(tmp)

                        elif self.pruning_method == 40:
                            #synflow pruning
                            tmp = (p.data.clone().type(torch.float) * (p.grad.clone().type(torch.float))).abs()
                            importance_scores.add_(tmp)

                        elif self.pruning_method == 2:
                            # magnitude
                            importance_scores.add_( p.data.clone().abs().type(torch.float) )

                        elif self.pruning_method == 1:
                            # random pruning
                            importance_scores =  p.data.clone().uniform_()
                        elif self.pruning_method == 6:
                            # squared gradient
                            importance_scores.add_((p.grad.data.type(torch.float)).detach().abs())
                    else:
                        #input_channels
                        # import pdb; pdb.set_trace()
                        summation_ch = [0,] + list(range(2,p.data.ndim))
                        summation_ch_view = [1, -1,] + [1,]*(p.data.ndim-2)

                        if self.pruning_method == 23:
                            tmp_score = (p.data.type(torch.float) * p.grad.data.type(torch.float)).detach().sum(summation_ch).pow(2).view(summation_ch_view)
                        elif self.pruning_method == 24:
                            # Taylor = abs(weight * gradient)
                            tmp_score = (p.data.type(torch.float) * p.grad.data.type(torch.float)).detach().sum(summation_ch).abs().view(summation_ch_view)
                        elif self.pruning_method == 25:
                            # Taylor = abs(weight * gradient)
                            tmp_score = (p.data.type(torch.float) * p.grad.data.type(torch.float)).detach().abs().sum(summation_ch).view(summation_ch_view)

                        elif self.pruning_method == 2:
                            # magnitude
                            tmp_score = p.data.clone().abs().type(torch.float).sum(summation_ch).view(summation_ch_view)
                            #importance_scores.add_(p.data.clone().abs().type(torch.float).sum(summation_ch).view(summation_ch_view))

                        importance_scores.add_(tmp_score.expand_as(importance_scores))

                    ##keep track of pruned
                    if 'pruning_mask' not in param_state:
                        param_state['pruning_mask'] = (importance_scores.clone()*0.0 + 1.0)

                pruning_mask = param_state['pruning_mask']

                # density_before = (float(pruning_mask.sum()) / float(pruning_mask.data.numel()))

                if do_pruning and (not self.is_finished):
                    # pruning iteration
                    # averaging scores between pruning iterations
                    ##do pruning
                    if self.sparsity_pattern in ["unstructured", "input_channels"]:
                        # pruning iteration
                        # averaging scores between pruning iterations
                        if 'importance_scores_tracked' not in param_state:
                            importance_scores_tracked = param_state[
                                'importance_scores_tracked'] = importance_scores.clone() * 0.0
                            importance_scores_tracked.add_(importance_scores)
                        else:
                            importance_scores_tracked = param_state['importance_scores_tracked']
                            importance_scores_tracked.mul_(momentum).add_(1. - momentum, importance_scores)

                        ##set to zero what was pruned already
                        importance_scores_tracked.mul_(pruning_mask)

                        sorted, indices = torch.sort(importance_scores_tracked.view(weight_size))
                        # Get pruning threshold

                        threshold_index = int(weight_size*pruning_ratio)

                        if not self.gglobal:
                            threshold_index = min(max(0, threshold_index), weight_size-1)
                            # keep_threshold = sorted[threshold_index]
                            # pdb.set_trace()
                            drop_indices = indices[:threshold_index]
                            # Remove those below threshold
                            if threshold_index > 0:
                                # Update pruning mask
                                pruning_mask.view(weight_size)[drop_indices] *= 0.0
                                # Easier:
                                # One potential bug is that we can remove more then required if there are more values <keep_threshold
                                #for the final step we will prune all below threshold. This might significantly reduce the parameters for Taylor pruning
                                # if (self.prune_last_with_threshold and (self.current_pruning_iteration==self.pruning_cycles)):
                                #     pruning_mask.mul_( (importance_scores_tracked > keep_threshold).type(pruning_mask.type()) )

                            print(
                                f"DEBUG: importance, stats: {importance_scores.min():8.2e}\t{importance_scores.max():8.2e}\t{importance_scores.mean():8.2e}\t{importance_scores.median():8.2e}"
                                f" density\t{float((importance_scores != 0.0).sum()) / float(importance_scores.numel()):3.2f}",
                                f" weight density\t{float((p.data != 0.0).sum()) / float(p.data.numel()):3.2f}",
                                f" pruning_mask density\t{float((pruning_mask.data != 0.0).sum()) / float(pruning_mask.data.numel()):3.2f}",
                                f" pruning_ratio\t{pruning_ratio:3.2f}")

                        else:
                            if self.thresholds is None:
                                self.thresholds = list()

                            for percentiles in range(0, weight_size, 128):
                                self.thresholds.append( sorted[min(percentiles, weight_size-1)])

                    elif self.sparsity_pattern == "ampere":

                        # importance_scores
                        if 1:
                            if torch.distributed.is_initialized():
                                #handle different GPUs having different stats:
                                torch.distributed.all_reduce(importance_scores.data)

                        if momentum>0.0:
                            if 'importance_scores_tracked' not in param_state:
                                importance_scores_tracked = param_state[
                                    'importance_scores_tracked'] = importance_scores.clone() * 0.0
                                importance_scores_tracked.add_(importance_scores)
                            else:
                                importance_scores_tracked = param_state['importance_scores_tracked']
                                importance_scores_tracked.mul_(momentum).add_(1. - momentum, importance_scores)

                            ##set to zero what was pruned already
                            importance_scores_tracked.mul_(pruning_mask.type(importance_scores_tracked.data.type()))
                            importance_scores.data = importance_scores_tracked.data.detach()

                        else:
                            #multiply importance score with pruning mask
                            importance_scores = importance_scores
                            # importance_scores.mul_(pruning_mask.type(importance_scores.data.type()))

                        if p.data.dim() == 2:
                            pruning_mask = mn_1d(importance_scores, m=4, n=self.pruning_level, dynamic_ratio = pruning_ratio)
                        elif p.data.dim() == 4:
                            #first layer in resnet50 is skipped
                            if p.data.shape[1] != 3:
                                shape = importance_scores.shape

                                x = importance_scores.permute(2, 3, 0, 1).contiguous().view(shape[2] * shape[3] * shape[0], shape[1], )

                                try:
                                    pruning_mask = m4n2_2d_best(x, density = 0.5, fraction=pruning_ratio)
                                except:
                                    print("skipping pruning for this layer with the shape : ", shape)

                                pruning_mask = pruning_mask.view(shape[2], shape[3], shape[0], shape[1]).permute(2,3,0,1).contiguous()
                                pruning_mask = pruning_mask.cuda()
                                param_state['pruning_mask'] = pruning_mask

                                if torch.distributed.is_initialized():
                                    torch.distributed.barrier()
                                    
                                if 1:
                                    p.data.mul_(pruning_mask.type(p.data.type()))
                                
                                
                                print(
                                    f"DEBUG: importance, stats: {importance_scores.min():8.2e}\t{importance_scores.max():8.2e}\t{importance_scores.mean():8.2e}\t{importance_scores.median():8.2e}"
                                    f" density\t{float((importance_scores != 0.0).sum()) / float(importance_scores.numel()):3.2f}",
                                    f" weight density\t{float((p.data != 0.0).sum()) / float(p.data.numel()):3.2f}",
                                    f" pruning_ratio\t{pruning_ratio:3.2f}")

                        threshold_index = 0
                        keep_threshold = 0

                if not self.gglobal:
                    # at some point because of CUDA, cudnn or pytorch we have to put this reference back:
                    param_state['pruning_mask'] = pruning_mask.cuda()

                    # zero out importance_scores
                    if not self.is_finished:
                        importance_scores.mul_(0.0)


                    # # make sure weights we pruned are zero
                    # p.data.mul_(pruning_mask.type(p.data.type()))
                    # # gradients to be safe as well
                    # p.grad.data.mul_(pruning_mask.type(p.grad.data.type()))

        if self.gglobal:
            # global pruning with single criteria for all layers
            if do_pruning and (not self.is_finished):
                total_neurons = 0
                non_zero_neurons = 0
                # import pdb; pdb.set_trace()
                global_pruning_thresholds = torch.tensor(self.thresholds)

                all_weight_size = global_pruning_thresholds.numel()

                sorted, indices = torch.sort(global_pruning_thresholds.view(-1))
                # Get pruning threshold

                threshold_index = int(all_weight_size * pruning_ratio)

                threshold_index = min(max(0, threshold_index), all_weight_size - 1)

                global_threshold = sorted[threshold_index]

                self.thresholds = list()

                layer = -1
                for group in self.param_groups:
                    momentum = group['momentum']

                    for p in group['params']:
                        if p.grad is None:
                            continue

                        layer += 1

                        if torch.isnan(p.grad.sum()):
                            print("PRUNING: skip because grad is NaN")
                            break

                        param_state = self.state[p]
                        if not self.is_finished:
                            weight_size = p.data.numel()

                            importance_scores_tracked = param_state['importance_scores_tracked']
                            pruning_mask = param_state['pruning_mask']
                            # import pdb; pdb.set_trace()
                            drop_indices = importance_scores_tracked.view(-1) < global_threshold.cuda()
                            # Remove those below threshold
                            if 1:
                                # Update pruning mask
                                pruning_mask.view(weight_size)[drop_indices] *= 0.0
                                # Easier:
                                # One potential bug is that we can remove more then required if there are more values <keep_threshold
                                # for the final step we will prune all below threshold. This might significantly reduce the parameters for Taylor pruning
                                # if (self.prune_last_with_threshold and (self.current_pruning_iteration==self.pruning_cycles)):
                                #     pruning_mask.mul_( (importance_scores_tracked > keep_threshold).type(pruning_mask.type()) )

                            print(
                                f"DEBUG: importance, stats: {importance_scores_tracked.min():8.2e}\t{importance_scores_tracked.max():8.2e}\t{importance_scores_tracked.mean():8.2e}\t{importance_scores_tracked.median():8.2e}"
                                f" density\t{float((importance_scores_tracked != 0.0).sum()) / float(importance_scores_tracked.numel()):3.2f}",
                                f" weight density\t{float((p.data != 0.0).sum()) / float(p.data.numel()):3.2f}",
                                f" pruning_mask density\t{float((pruning_mask.data != 0.0).sum()) / float(pruning_mask.data.numel()):3.2f}",
                                f" pruning_ratio\t{pruning_ratio:3.2f}")

                            # at some point because of CUDA, cudnn or pytorch we have to put this reference back:
                            param_state['pruning_mask'] = pruning_mask.cuda()

                            # zero out importance_scores
                            importance_scores.mul_(0.0)

        total_neurons = 0
        non_zero_neurons = 0
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                # if torch.isnan(p.grad.sum()):
                #     print("PRUNING: skip because grad is NaN, calc stats")
                #     break 

                param_state = self.state[p]
                pruning_mask = param_state['pruning_mask']
                # make sure weights we pruned are zero
                p.data.mul_(pruning_mask.type(p.data.type()))
                # gradients to be safe as well
                if not(p.grad is None):
                    p.grad.data.mul_(pruning_mask.type(p.grad.data.type()))

                total_neurons += p.data.numel()
                non_zero_neurons += (p.data!=0).sum()

        if do_pruning:
            # import pdb; pdb.set_trace()
            print("pruning results", total_neurons, "/", non_zero_neurons.item(), "{:3.2}".format(float(non_zero_neurons.item())/float(total_neurons)))


        return loss

    def enforce(self):
        """Enforces pruned pattern
        """
        layer=-1
        for group in self.param_groups:
            for p in group['params']:
                layer += 1
                param_state = self.state[p]

                # at first step we try to kill all buffers:
                if 'importance_scores_tracked' in param_state:
                    del param_state['importance_scores_tracked']

                if 'importance_scores' in param_state:
                    del param_state['importance_scores']

                if 'pruning_mask' not in param_state:
                    param_state['pruning_mask'] = p.data.clone() * 0.0 + 1.0
                    print(f"creating pruning_mask,  {param_state['pruning_mask'].type()}")

                if p.data.type() != param_state['pruning_mask'].type():
                    param_state['pruning_mask'] = param_state['pruning_mask'].type(p.data.type())
                    print(f"changing now:,  {param_state['pruning_mask'].type()}")

                pruning_mask = param_state['pruning_mask']

                    # make sure weights we pruned are zero
                p.data.mul_(pruning_mask)

                # gradients to be safe as well
                if not (p.grad is None):
                    p.grad.data.mul_(pruning_mask.type(p.grad.data.type()))

    def add_noise(self, std=0.01):
        """add_noise
        """
        self.total_neurons = 0
        self.total_zero = 0
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                noise = std * p.data.clone().normal_().cuda()

                if 0:
                    importance_scores = param_state['importance_scores']
                    ranking = torch.argsort(importance_scores.view(-1)).view(importance_scores.shape)
                    ranking = ranking.type(p.data.type())
                    ranking = ranking/ranking.max()
                    # ranking = 1.0 - ranking
                    # import pdb; pdb.set_trace()
                    noise = noise * ranking

                    param_state['noise'] = noise
                prob = 0.95
                if 1:
                    noise = p.data.clone().bernoulli_(p=prob).cuda()
                    # noise = p.data.clone().bernoulli_(p=0.8).cuda()*0.5 + 0.5
                    # import pdb; pdb.set_trace()

                    param_state['noise'] = noise



                param_state['orig'] = p.data.clone()

                # print(p.data.std(), noise.std())

                # p.data = p.data + noise
                p.data = p.data * noise

                self.total_neurons += noise.numel()
                self.total_zero += (1-prob)*self.total_neurons
                # print(self.total_zero)

        self.use_noise = True

    def remove_noise(self):
        """doing pruning
        """

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                # weight_size = p.data.numel()
                if 0:
                    if 'noise' in param_state:
                        noise = param_state['noise']

                    p.data = p.data - noise
                else:
                    p.data = param_state['orig'].clone().cuda()

        self.use_noise = False


    def weights_to_abs(self, model):
        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        self.signs = linearize(model)

    def weights_to_normal(self, model):
        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        nonlinearize(model, self.signs)

