import matplotlib.pyplot as plt
import numpy as np
import sys

path = ""
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
        if not got_name:
            layer_names = []

        if "compute_criteria_from" in line:
            if "bias" not in line:
                got_name = True
                layer_name = line.split(" ")[1][:-1].split(":")[0]
                layer_names.append(layer_name)

        if "pruned_perc" in line:
            got_name = False
            pruned = line.split(" ")[1][1:-1]
            total = line.split(" ")[2][:-2]
            for lni, layer_name in enumerate(layer_names):
                # if lni > 0:
                #     continue
                # if "head." in layer_name:
                #     continue
                pruning_list.append({})
                pruning_list[-1]["layer_name"] = layer_name.replace("backbone._operations.","").replace(".weight","")
                pruning_list[-1]["layer_name"] = pruning_list[-1]["layer_name"].replace("conv_operation.1","conv_bn")
                # pruning_list[-1]["layer_name"] = pruning_list[-1]["layer_name"].replace("last_bn_layer","bn")
                pruning_list[-1]["layer_name_id"] = int(layer_name.split(".")[2])
                if "head" in layer_name:
                    pruning_list[-1]["layer_name_id"] = pruning_list[-1]["layer_name_id"] + 20
                pruning_list[-1]["pruned"] = int(pruned)
                pruning_list[-1]["total"] = int(total)


#start ploting
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot()

#sort by layer name
pruning_list.sort(key = lambda x: x["layer_name_id"])

pruned_all = [a["total"]-a["pruned"] for a in pruning_list]

pruned_all_total = [a["total"] for a in pruning_list]

barWidth = 0.75

r1 = np.arange(len(pruned_all))

plt.bar(r1, pruned_all_total, color='darkorange', width=barWidth, edgecolor='grey', label='_nolegend_', alpha=0.3)

# Make the plot
plt.bar(r1, pruned_all, color='forestgreen', width=barWidth, edgecolor='black', label='mlp.dense_h_to_4h_')

plt.xticks([r for r in range(len(pruned_all))], [pruning_list[layer_id]["layer_name"] for layer_id in range(len(pruned_all))], rotation='vertical')
plt.subplots_adjust(bottom=0.35)

line_height = 20
vertical_start = max(pruned_all_total) + line_height + 5
for i, v in enumerate(pruned_all):
    plt.text(r1[i], vertical_start, str(v), color='black', fontsize=7, horizontalalignment="center") #, fontweight='bold')
    plt.text(r1[i], vertical_start + line_height, str(pruned_all_total[i]), color='black', fontsize=7, horizontalalignment="center") #, fontweight='bold')

if 1:
    plt.text(-1, vertical_start + line_height, "Max:", color='black', fontsize=7, horizontalalignment="center")  # , fontweight='bold')
    plt.text(-1, vertical_start, "Remain: ", color='black', fontsize=7, horizontalalignment="center")  # , fontweight='bold')
# Create legend & Show graphic
# plt.legend(loc='upper center', bbox_to_anchor=(0.62, -0.1), ncol=3)
# plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1), ncol=3)
plt.savefig("arch_prune10.png", bbox_inches = 'tight')
plt.show()
[print(a["layer_name"]) for a in pruning_list]
