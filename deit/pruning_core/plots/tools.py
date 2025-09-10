import glob
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from copy import deepcopy as copy

def plot_all_criteria(array_of_oracle, num_layers=16, prefix = ""):
    fig_all = plt.figure(figsize=(12.0, 6.0))
    plt.cla()
    plt.clf()
    ax = plt.subplot(111)
    total_x = 0
    data_structure_all = list()
    for layer in range(num_layers):
        oracle_values = np.asarray(array_of_oracle[layer])

        # oracle_values = np.random.uniform(1.0, 2.0, size = oracle_values.shape)
        oracle_values_mean = oracle_values.mean(axis=0)
        oracle_values_mean = np.median(oracle_values, axis=0)
        oracle_values_std = oracle_values.std(axis=0) * 0.0
        oracle_values_std = oracle_values.std(axis=0)
        oracle_values_std =  np.stack(( oracle_values_mean - oracle_values.min(axis=0), oracle_values.max(axis=0) - oracle_values_mean))


        ##remove lines
        oracle_values_std = oracle_values_mean
        oracle_values_std = oracle_values_mean
        oracle_values_std =  np.stack(( oracle_values_mean*0.0, 0.0*oracle_values_mean))

        try:
            total_x += len(oracle_values_mean)
        except:
            import pdb; pdb.set_trace()

        for el in range(len(oracle_values_mean)):
            data_structure_all.append({'x': el, 'y': oracle_values_mean[el], 'std': oracle_values_std[:,el], 'layer': layer})

    # import pdb; pdb.set_trace()
    ##sort values:
    sort_function = lambda x: x['y']
    data_structure_all.sort(key = sort_function)

    import pdb; pdb.set_trace()

    start_value = 0
    for layer in range(num_layers):

        oracle_values_mean = [d['y'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = [d['std'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = np.stack(oracle_values_std).T
        x = [ind for ind,d in enumerate(data_structure_all) if layer == d['layer']]

        x = [d['x'] for d in data_structure_all if layer == d['layer']]

        if 1:
            ##if sort per each layer:
            oracle_values_mean = np.asarray(oracle_values_mean)
            sort_indx = np.argsort(-oracle_values_mean)
            oracle_values_mean= oracle_values_mean[sort_indx]
            oracle_values_std= oracle_values_std[:,sort_indx]
            x = range(len(oracle_values_mean))

        # import pdb; pdb.set_trace()
        ax.errorbar(x, oracle_values_mean, yerr=oracle_values_std, fmt='o', label = "layer %d"%layer)

        plt.title(['abs(loss) - per layer', ])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right=0.7)
        ax.set_yscale("log")
        plt.grid(True)

    fig_all.savefig('images/all_layers_%s.png'%prefix)
    fig_all = plt.figure(figsize=(16.0, 5.0))

    plt.cla();plt.clf()
    ax = plt.subplot(111)
    tot_x = 0
    for layer in range(num_layers):

        oracle_values_mean = [d['y'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = [d['std'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = np.stack(oracle_values_std).T
        x = [ind for ind,d in enumerate(data_structure_all) if layer == d['layer']]

        x = [d['x'] for d in data_structure_all if layer == d['layer']]


        if 1:
            ##if sort per each layer:
            oracle_values_mean = np.asarray(oracle_values_mean)
            sort_indx = np.argsort(-oracle_values_mean)
            oracle_values_mean= oracle_values_mean[sort_indx]
            oracle_values_std= oracle_values_std[:,sort_indx]
            x = tot_x + np.asarray(range(len(oracle_values_mean)))
            tot_x += len(oracle_values_mean)

        # import pdb; pdb.set_trace()
        # ax.errorbar(x, oracle_values_mean, yerr=oracle_values_std, fmt='o', label = "layer %d"%layer)
        ax.errorbar(x, oracle_values_mean, fmt='o', label = "layer %d"%layer)

        plt.title(['abs(loss) - per layer', ])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right=0.7)
        ax.set_yscale("log")
        plt.grid(True)

    fig_all.savefig('images/all_layers_2_%s.png'%prefix)


def plot_all_criteria_crit(array_of_oracle, num_layers=16, prefix = ""):

    total_x = 0
    data_structure_all = list()
    for layer in range(num_layers):
        oracle_values = np.asarray(array_of_oracle[layer])

        # oracle_values = np.random.uniform(1.0, 2.0, size = oracle_values.shape)
        oracle_values_mean = oracle_values
        oracle_values_std = oracle_values
        oracle_values_std = oracle_values

        oracle_values_std =  np.stack(( oracle_values_mean - oracle_values.min(axis=0), oracle_values.max(axis=0) - oracle_values_mean))
        try:
            total_x += len(oracle_values_mean)
        except:
            import pdb; pdb.set_trace()

        for el in range(len(oracle_values_mean)):
            data_structure_all.append({'x': el, 'y': oracle_values_mean[el], 'std': oracle_values_std[:,el], 'layer': layer})

    # import pdb; pdb.set_trace()
    ##sort values:
    sort_function = lambda x: x['y']
    data_structure_all.sort(key = sort_function)

    # import pdb; pdb.set_trace()

    ##create a figure with all cummulative importance
    fig_all = plt.figure(figsize=(12.0, 6.0))
    plt.cla()
    plt.clf()
    ax = plt.subplot(111)

    contribution = [d['y'] for d in data_structure_all]
    x = range(len(contribution))
    ax.plot(x, contribution)
    plt.yscale('log')
    plt.grid(True)

    fig_all.savefig('images/cumulative_%s.png' % prefix)


    fig_all = plt.figure(figsize=(12.0, 6.0))
    plt.cla()
    plt.clf()
    ax = plt.subplot(111)
    start_value = 0
    for layer in range(num_layers):

        oracle_values_mean = [d['y'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = [d['std'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = np.stack(oracle_values_std).T
        x = [ind for ind,d in enumerate(data_structure_all) if layer == d['layer']]

        x = [d['x'] for d in data_structure_all if layer == d['layer']]

        if 1:
            ##if sort per each layer:
            oracle_values_mean = np.asarray(oracle_values_mean)
            sort_indx = np.argsort(-oracle_values_mean)
            oracle_values_mean= oracle_values_mean[sort_indx]
            oracle_values_std= oracle_values_std[:,sort_indx]
            x = range(len(oracle_values_mean))

        # import pdb; pdb.set_trace()
        ax.errorbar(x, oracle_values_mean, fmt='o', label = "layer %d"%layer)

        plt.title(['abs(loss) - per layer', ])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right=0.7)
        ax.set_yscale("log")
        plt.grid(True)

    fig_all.savefig('images/all_layers_%s.png'%prefix)
    fig_all = plt.figure(figsize=(16.0, 5.0))

    plt.cla();plt.clf()
    ax = plt.subplot(111)
    tot_x = 0
    for layer in range(num_layers):

        oracle_values_mean = [d['y'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = [d['std'] for d in data_structure_all if layer == d['layer']]
        oracle_values_std = np.stack(oracle_values_std).T
        x = [ind for ind,d in enumerate(data_structure_all) if layer == d['layer']]

        x = [d['x'] for d in data_structure_all if layer == d['layer']]


        if 1:
            ##if sort per each layer:
            oracle_values_mean = np.asarray(oracle_values_mean)
            sort_indx = np.argsort(-oracle_values_mean)
            oracle_values_mean= oracle_values_mean[sort_indx]
            oracle_values_std= oracle_values_std[:,sort_indx]
            x = tot_x + np.asarray(range(len(oracle_values_mean)))
            tot_x += len(oracle_values_mean)

        # import pdb; pdb.set_trace()
        # ax.errorbar(x, oracle_values_mean, yerr=oracle_values_std, fmt='o', label = "layer %d"%layer)
        ax.errorbar(x, oracle_values_mean, fmt='o', label = "layer %d"%layer)

        plt.title(['abs(loss) - per layer', ])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right=0.7)
        ax.set_yscale("log")
        plt.grid(True)

    fig_all.savefig('images/all_layers_2_%s.png'%prefix)


def plot_per_layer_joint(oracle_per_layer, criteria_per_layer, active_layers, file_suffix, normalized = False):
    fig = plt.figure()
    for layer in range(len(criteria_per_layer)):
        plt.cla()
        plt.clf()

        criteria_to_draw = criteria_per_layer[layer]
        oracle_to_draw   = oracle_per_layer[layer]

        if normalized:
            criteria_to_draw = criteria_to_draw/np.linalg.norm(criteria_to_draw, 2)
            oracle_to_draw = oracle_to_draw/np.linalg.norm(oracle_to_draw, 2)

        plt.stem(oracle_to_draw)
        # plt.stem(-(criteria_to_draw),'g')
        markerline, stemlines, baseline = plt.stem(-criteria_to_draw, 'g')
        plt.setp(markerline, 'markerfacecolor', 'g')

        plt.legend(['oracle', 'criteria'])
        fig.savefig('images/{}layer_{}.png'.format(file_suffix, active_layers[layer]))

def compute_ranking_correlation(criteria_per_layer, oracle_per_layer, noramlize):
    fig = plt.figure()
    plt.cla()
    plt.clf()
    for el in range(len(criteria_per_layer)):
        if noramlize:
            criteria_per_layer[el] = criteria_per_layer[el] / np.linalg.norm(criteria_per_layer[el], norm_choose)

        if el == 0:
            criteria_per_layer_full = criteria_per_layer[el]
        else:
            criteria_per_layer_full = np.hstack((criteria_per_layer_full,criteria_per_layer[el]))

    for el in range(len(oracle_per_layer)):
        if noramlize:
            oracle_per_layer[el] = oracle_per_layer[el] / np.linalg.norm(oracle_per_layer[el], norm_choose)

        if el == 0:
            oracle_per_layer_full = oracle_per_layer[el]
        else:
            oracle_per_layer_full = np.hstack((oracle_per_layer_full,oracle_per_layer[el]))


    spearman_full = compute_criteria(oracle_per_layer_full, criteria_per_layer_full)

    return spearman_full

def compute_criteria(x,y):
    #x large array
    #y is a subset
    def compute_rank(x):
        order = (-x).argsort()
        ranks = np.empty(len(x), int)
        try:
            ranks[order] = np.arange(len(x))
        except:
            pdb.set_trace()

        return ranks, order

    ranks_x, order_x = compute_rank(x)
    ranks_y, _ = compute_rank(y)

    weight = np.zeros(len(ranks_x), float)
    weight_full = np.linspace(0, 1.0, ranks_x.size)

    SCALE = 1.0

    weight[order_x] = 1.0 - weight_full
    if SCALE == 1.0:
        weight[order_x] = 1.0
    #weight = weight / (weight**2).sum()

    d2 = ((np.asarray(ranks_x,dtype = np.float) - np.asarray(ranks_y, dtype = np.float)) * weight)**2

    n = ranks_x.size

    spearmanr = 1 - SCALE*6*d2.sum()/(n**3 - n)
    return spearmanr

def compute_avg_rank(x,y):
    #x large array
    #y is a subset

    temp = (-x).argsort()
    ranks = np.empty(len(x), int)
    ranks[temp] = np.arange(len(x))

    loc_ranks = list()
    for el in y:
        loc_ranks.append(np.mean(ranks[el == x]))

    loc_ranks = np.asarray(loc_ranks)

    return loc_ranks

def get_global_ranking_statistics(criteria_per_layer, noramlize, file_suffix):



    fig = plt.figure()
    plt.cla()
    plt.clf()
    for el in range(len(criteria_per_layer)):
        if noramlize:
            criteria_per_layer[el] = criteria_per_layer[el] / np.linalg.norm(criteria_per_layer[el], norm_choose)

        if el == 0:
            criteria_per_layer_full = criteria_per_layer[el]
        else:
            criteria_per_layer_full = np.hstack((criteria_per_layer_full,criteria_per_layer[el]))

    medians_criteria = list()
    mins_criteria = list()
    maxs_criteria = list()

    ##get statistics of min,max and median
    for el in range(len(criteria_per_layer)):
        ranking = compute_avg_rank(criteria_per_layer_full,criteria_per_layer[el])

        if 1:
            medians_criteria.append(np.mean(ranking))
            mins_criteria.append(np.min(ranking))
            maxs_criteria.append(np.max(ranking))
        else:
            medians_criteria.append(np.mean(ranking))
            mins_criteria.append(np.mean(ranking) - np.std(ranking))
            maxs_criteria.append(np.mean(ranking) + np.std(ranking))

    plt.cla()
    plt.clf()
    xplot=range(len(criteria_per_layer))
    plt.plot(xplot,medians_criteria, marker='.')
    plt.plot(xplot,mins_criteria,'b--')
    plt.plot(xplot,maxs_criteria,'b--')
    plt.xlabel('Layer #')
    plt.ylabel('Rank over all neurons')
    plt.title('After pruning')
    plt.title('Before pruning')

    plt.legend(['median', 'min', 'max'])
    # plt.legend(['mean', '- std', '+ std'])
    fig.savefig('images/global_ranking_statistics_{}.png'.format(file_suffix))

    return 0

def get_global_statistics(criteria_per_layer, noramlize, file_suffix):
    fig = plt.figure()
    plt.cla()
    plt.clf()
    for el in range(len(criteria_per_layer)):
        if noramlize:
            criteria_per_layer[el] = criteria_per_layer[el] / np.linalg.norm(criteria_per_layer[el], norm_choose)

        if el == 0:
            criteria_per_layer_full = criteria_per_layer[el]
        else:
            criteria_per_layer_full = np.hstack((criteria_per_layer_full,criteria_per_layer[el]))

    medians_criteria = list()
    mins_criteria = list()
    maxs_criteria = list()

    ##get statistics of min,max and median
    for el in range(len(criteria_per_layer)):
        ranking = criteria_per_layer[el]
        medians_criteria.append(np.mean(ranking))
        # medians_criteria.append(np.mean(ranking))
        mins_criteria.append(np.min(ranking))
        maxs_criteria.append(np.max(ranking))

    plt.cla()
    plt.clf()
    xplot=range(len(criteria_per_layer))
    plt.plot(xplot,medians_criteria, marker='.')
    plt.plot(xplot,mins_criteria,'b--')
    plt.plot(xplot,maxs_criteria,'b--')

    plt.legend(['median', 'min', 'max'])
    fig.savefig('images/global_statistics_{}.png'.format(file_suffix))


    return 0


def compute_stats(oracle_per_layer, criteria_per_layer):
    def get_corr_func(a, b):
        from scipy.stats import kendalltau, spearmanr

        def _pearson(a, b):
            return np.corrcoef(a, b)[0, 1]

        def _kendall(a, b):
            rs = kendalltau(a, b)
            if isinstance(rs, tuple):
                return rs[0]
            return rs

        def _spearman(a, b):
            return spearmanr(a, b)[0]

        _cor_methods = {
            'pearson': _pearson(a,b),
            'kendall': _kendall(a,b),
            'spearman': _spearman(a,b)
        }
        return _cor_methods

    # del oracle_per_layer[-1]
    # del oracle_per_layer[-1]

    correlation_list = {'spearman': list(), 'pearson': list(), 'kendall': list()}
    for layer in range(len(criteria_per_layer)):
        criteria_layer = copy(criteria_per_layer[layer])
        oracle_layer = oracle_per_layer[layer]
        criteria_layer = criteria_layer[:len(oracle_layer)]

        correlations = get_corr_func(oracle_layer, criteria_layer)

        for key in correlations.keys():
            correlation_list[key].append(correlations[key])

        print("layer {} per layer:\t".format(layer), "\t Pearson:", "\t{:10s}".format("%2.4f" % correlations['pearson']), "\t SC:",
              "\t{:10s}".format("%2.4f" % correlations['spearman']), "\t KT:", "\t{:10s}".format("%2.4f" % correlations['kendall']))

    spearman = np.mean(np.asarray([a for a in correlation_list['spearman']]))
    pearson = np.mean(np.asarray([a for a in correlation_list['pearson']]))
    kendall = np.mean(np.asarray([a for a in correlation_list['kendall']]))

    print("Mean per layer:\t", "\t Pearson:", "\t{:10s}".format("%2.4f"%pearson),  "\t SC:", "\t{:10s}".format("%2.4f"%spearman), "\t KT:", "\t{:10s}".format("%2.4f"%kendall))

    criteria_per_layer_full = criteria_per_layer[0].copy()
    for da in criteria_per_layer[1:]:
        criteria_per_layer_full = np.hstack((criteria_per_layer_full, da))

    oracle_per_layer_full = oracle_per_layer[0].copy()
    for da in oracle_per_layer[1:]:
        oracle_per_layer_full = np.hstack((oracle_per_layer_full, da))

    # import pdb; pdb.set_trace()
    correlations = get_corr_func(oracle_per_layer_full, criteria_per_layer_full)

    print("Over all layers: ", "\t Pearson:", "\t{:10s}".format("%2.4f" % correlations['pearson']), "\t SC:",
          "\t{:10s}".format("%2.4f" % correlations['spearman']), "\t KT:",
          "\t{:10s}".format("%2.4f" % correlations['kendall']))

    #hit over percantages

    oracle_per_layer_full_ranks = oracle_per_layer_full.argsort()
    criteria_per_layer_full_ranks = criteria_per_layer_full.argsort()
    tot_length = len(oracle_per_layer_full)

    percentiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    overlap_prob_dict = {}
    for percentile in percentiles:
        oracle_selection = oracle_per_layer_full_ranks[int(tot_length*percentile):]
        criteria_selection = criteria_per_layer_full_ranks[int(tot_length*percentile):]

        overlap_prob_dict[percentile] = np.in1d(oracle_selection, criteria_selection, assume_unique=True).mean()

    # import pdb;pdb.set_trace()
    string = "Overlap percentiles: \n" + " ".join(["{}p : {:3.2f} \t".format(key, value) for key, value in overlap_prob_dict.items()])
    print(string)
