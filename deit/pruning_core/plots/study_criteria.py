import glob
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from copy import deepcopy as copy
from tools import plot_per_layer_joint, compute_ranking_correlation, compute_criteria, compute_avg_rank, get_global_ranking_statistics, get_global_statistics, plot_all_criteria, plot_all_criteria_crit



path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/temp/debug/criteria_*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/oracle_bn_rep2/debug/criteria_*"
# path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/oracle_nobn_rep2/debug/criteria_*"
# path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/runs/pruning_oracle22_3/debug/criteria_*"
# path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/runs/pruning_m22_noentropy_amp_train_conv/debug/criteria_*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/runs/pruning_m22_noentropy_amp_train_conv_big/debug/criteria_*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/NeMo/runs/pruning_test_skip_nobn7_withbn_skipcon/debug/criteria_*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/superres/test/debug/criteria_*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/superres/test_prune_adamw_3_2000_2/debug/criteria_*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/oracle/oracle_temp_bn_*.p"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/compute_criteria23/debug/crite*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/oracle_collect2/debug/crite*"
path_to_criterias = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/criteria_collect23/debug/crite*"

# oracle_nobn_rep2
import glob

files_there = sorted(glob.glob(path_to_criterias))

for cf_ind, criteria_file in enumerate(files_there):
    plt.close('all')
    print(criteria_file)
    after_pruning = "after" in criteria_file
    postfix = "_after" if after_pruning else "_before"

    after_pruning = False

    criteria_per_layer = pickle.load(open(criteria_file, "rb"))
    if 0:
        #only for oracle with interm compute
        criteria_per_layer = criteria_per_layer['loss_list']
        criteria_per_layer = [np.asarray(a) for a in criteria_per_layer]
        pickle.dump(criteria_per_layer, open("oracle_nas.p", "wb"))
    # import pdb; pdb.set_trace()

    if after_pruning:
        ###
        ###    ONLY FOR PRUNED MODEL
        ###
        ##remove 0 criteria, useful for pruned network only
        for el in range(len(criteria_per_layer)):
            non_zero_coef = np.nonzero(criteria_per_layer[el])[0]
            criteria_per_layer[el] = criteria_per_layer[el][non_zero_coef]
            # oracle_per_layer[el] = oracle_per_layer[el][non_zero_coef]

    if 1:
        thr = 1e-16
        for el in range(len(criteria_per_layer)):
            criteria_per_layer[el][criteria_per_layer[el] < thr] = thr


    array_of_oracle = criteria_per_layer
    num_layers = len(array_of_oracle)

    # import pdb; pdb.set_trace()

    if 0:
        ##go over layers and show mean and std
        fig = plt.figure()
        for layer in range(num_layers):
            oracle_values = np.asarray(array_of_oracle[layer])

            # import pdb ;pdb.set_trace()
            # oracle_values = np.random.uniform(1.0, 2.0, size = oracle_values.shape)
            # oracle_values_mean = oracle_values.mean(axis=0)
            # # oracle_values_mean = np.median(oracle_values, axis=0)
            # oracle_values_std = oracle_values.std(axis=0) * 0.0
            # oracle_values_std = oracle_values.std(axis=0)

            # oracle_values_std =  np.stack(( oracle_values_mean - oracle_values.min(axis=0), oracle_values.max(axis=0) - oracle_values_mean))

            x = range(len(oracle_values))
            plt.cla()
            plt.clf()
            ax = plt.subplot(111)

            sorting_order = np.argsort(-oracle_values)

            ##sort
            # oracle_values_mean = oracle_values_mean[sorting_order]
            # oracle_values_std = oracle_values_std[:,sorting_order]

            ax.errorbar(x, oracle_values, fmt='o')

            plt.legend(['layer %d'%layer, ])
            # ax.set_yscale("log")
            plt.grid(True)
            fig.savefig('images/it%02d_layer_%03d.png'%(cf_ind, layer))


    if 1:
        ##draw all_of them
        plot_all_criteria_crit(array_of_oracle, num_layers = num_layers, prefix = "it%02d"%cf_ind)


    if 1:
        full_array = list()
        for layer in range(num_layers):
            full_array.append(np.asarray(array_of_oracle[layer]))



        get_global_ranking_statistics(full_array, noramlize = False, file_suffix="it%02d"%cf_ind)
        get_global_statistics(full_array, noramlize = False, file_suffix="it%02d"%cf_ind)
#
# if 1:
#     oracle_per_layer = list()
#     for layer in range(num_layers):
#         oracle_per_layer.append(np.asarray(array_of_oracle[layer]).mean(axis=0))
#
#     with open("oracle.pickle","wb") as f:
#         pickle.dump(oracle_per_layer, f)
#
#     import pickle
#     filename="/home/scratch.pmolchanov_nvresearch/PRUNING_project/pytorch/dlaa-pytorch-resnet/fresh/dlaa-pytorch/pruning/study/oracles/oracle_simple_folded.pickle"
#     with open(filename,"rb") as f:
#         oracle_per_layer = pickle.load(f)
#
#     len(oracle_per_layer)
#     len(oracle_per_layer[0])
# import pdb; pdb.set_trace()
