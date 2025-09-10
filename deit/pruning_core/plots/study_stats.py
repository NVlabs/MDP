import glob
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from copy import deepcopy as copy
from tools import plot_per_layer_joint, compute_ranking_correlation, compute_criteria, compute_avg_rank, get_global_ranking_statistics, get_global_statistics, plot_all_criteria, plot_all_criteria_crit

from tools import compute_stats

# import pdb; pdb.set_trace()
# with open("jasper/oracle.pickle", "rb") as f:
# with open("oracle_nas.p", "rb") as f:
oracle_path = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/oracle_collect2/criteria_40.pickle"
with open(oracle_path, "rb") as f:
    oracle = pickle.load(f)


# path_to_criteria = "../../runs/jasper/test_structure_bn30_oracle/criteria_40.pickle"
# path_to_criteria = "../../runs/jasper/test_structure_bn30_oracle/criteria_40.pickle"
# path_to_criteria = "../../runs/jasper/cluster/new_data3/structure_group8_m23_bn1k_m0_it_10000_wd_1e-6_lr_10.0_pruning_0.3/criteria_23.pickle"
# path_to_criteria = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/compute_criteria23/criteria_23.pickle"
path_to_criteria = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/criteria_collect23/criteria_23.pickle"
path_to_criteria = "/media/pmolchanov/Ubuntu18_work/PROJECTS/PRUNING/nvPruner/runs/bodypose_nas/criteria_collect23/criteria_22.pickle"

with open(path_to_criteria, "rb") as f:
    store_criteria = pickle.load(f)

compute_stats(oracle, store_criteria)

