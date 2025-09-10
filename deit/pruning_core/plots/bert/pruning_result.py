import matplotlib.pyplot as plt
import numpy as np
import sys

path = "./bs4_group8_m23_m09_klc_100000.0_hlc_0.0_olc_0.0_pp_0.65_seval_True_klct_100.0_hlct_0.0_olct_0.0_lr_1e-05_debugOutput_pruning_00000123.txt"
if len(sys.argv) > 1:
    path = sys.argv[1]

if not path.endswith(".txt"):
    import glob
    all_files = sorted(glob.glob(path + "/debug/*.txt"))
    if len(all_files) == 0:
        print("No debug files found, exit")
        exit()
    path = all_files[-1]

got_name = False
pruning_list = list()
with open(path, "r") as f:
    for line in f:
        if "compute_criteria_from" in line:
            got_name = True
            layer_name = line.split(" ")[1][:-1]

        if "pruned_perc" in line:
            got_name = False
            pruned = line.split(" ")[1][1:-1]
            total = line.split(" ")[2][:-2]
            pruning_list.append({})
            pruning_list[-1]["layer_name"] = layer_name
            pruning_list[-1]["pruned"] = int(pruned)
            pruning_list[-1]["total"] = int(total)


#start ploting
fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot()

pruned_att = [a["total"]-a["pruned"] for a in pruning_list[0::3]]
pruned_att_dense = [a["total"]-a["pruned"] for a in pruning_list[1::3]]
pruned_mlp = [a["total"]-a["pruned"] for a in pruning_list[2::3]]

pruned_att_total = [a["total"] for a in pruning_list[0::3]]
pruned_att_dense_total = [a["total"] for a in pruning_list[1::3]]
pruned_mlp_total = [a["total"] for a in pruning_list[2::3]]

barWidth = 0.25

r1 = np.arange(len(pruned_att))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, pruned_att_total, color='darkorange', width=barWidth, edgecolor='grey', label='_nolegend_', alpha=0.2)
plt.bar(r2, pruned_att_dense_total, color='royalblue', width=barWidth, edgecolor='grey', label='_nolegend_', alpha=0.2)
plt.bar(r3, pruned_mlp_total, color='forestgreen', width=barWidth, edgecolor='grey', label='_nolegend_', alpha=0.2)

# Make the plot
plt.bar(r1, pruned_att, color='darkorange', width=barWidth, edgecolor='black', label='attention.query_key_value_')
plt.bar(r2, pruned_att_dense, color='royalblue', width=barWidth, edgecolor='black', label='attention.dense_')
plt.bar(r3, pruned_mlp, color='forestgreen', width=barWidth, edgecolor='black', label='mlp.dense_h_to_4h_')

# Add xticks on the middle of the group bars
plt.xlabel('BERT block:', horizontalalignment='left', x=0.0)
# plt.get_xaxis.set_label_coords(0.05, -0.025)
plt.xticks([r + barWidth for r in range(len(pruned_att))], ["%d"%layer_id for layer_id in range(len(pruned_att))])

vertical_start = 4350
for i, v in enumerate(pruned_att):
    plt.text(r2[i], vertical_start + 2*225, str(v), color='black', fontsize=7, horizontalalignment="center") #, fontweight='bold')
    plt.text(r2[i], vertical_start + 225, str(pruned_att_dense[i]), color='black', fontsize=7, horizontalalignment="center") #, fontweight='bold')
    plt.text(r2[i], vertical_start, str(pruned_mlp[i]), color='black', fontsize=7, horizontalalignment="center") #, fontweight='bold')


if 1:
    plt.text(-1, vertical_start + 225*3+10, "Max:", color='black', fontsize=7, horizontalalignment="center")  # , fontweight='bold')
    plt.text(-1, vertical_start + 225*2, "3072", color='black', fontsize=7, horizontalalignment="center")  # , fontweight='bold')
    plt.text(-1, vertical_start + 225, "1024", color='black', fontsize=7, horizontalalignment="center")  # , fontweight='bold')
    plt.text(-1, vertical_start, "4096", color='black', fontsize=7, horizontalalignment="center")  # , fontweight='bold')
# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.62, -0.1), ncol=3)
# plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1), ncol=3)
plt.savefig("image.png", bbox_inches = 'tight')
plt.show()
