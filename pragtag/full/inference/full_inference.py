import json
from collections import Counter
import numpy as np

'''
######################
Source code for majority labeling test data
######################
'''

all_predictions = []
seeds = ['1_rec_30.json', '2_rec_50.json', '3_majority.json', '4_synonym.json', '5_rec_nosym.json', '6_test.json']

for address in seeds:
    with open("tags/"+address, "r") as t1:
        temp = json.load(t1)
        all_predictions.append(temp)

#### If tie happens, stick to label of highest recall
order = {'Strength':0, 'Weakness':1, 'Structure':2, 'Recap':3, 'Todo':4, 'Other':5}
recall = [
    [0.9032258064516129, 0.8513513513513513, 1.0, 0.8840579710144928, 0.9797979797979798, 0.6413043478260869],
    [0.8387096774193549, 0.8783783783783784, 1.0, 0.8985507246376812, 0.9797979797979798, 0.6304347826086957],
    [0.8709677419354839, 0.8918918918918919, 0.9772727272727273, 0.855072463768116, 0.9595959595959596, 0.6304347826086957],
    [0.8387096774193549, 0.8648648648648649, 1.0, 0.8260869565217391, 0.98989898989899, 0.6739130434782609],
    [0.9032258064516129, 0.8378378378378378, 1.0, 0.8115942028985508, 0.9696969696969697, 0.6521739130434783],
    [0, 0, 0, 0, 0, 0]
]

#### Majority label
final_predictions = []
for i in range(len(all_predictions[0])):
    now_dict = {"id": all_predictions[0][i]["id"]}
    lab = []
    for j in range(len(all_predictions[0][i]["labels"])):
        labs = [all_predictions[k][i]["labels"][j] for k in range(len(seeds))]
        mc = Counter(labs).most_common(len(seeds))
        same = [mc[0]]
        for c in mc[1:]:
            if same[0][1] == c[1]: same.append(c)
            else: break
        if len(same) >= 1:
            mx = [0 for _ in range(len(same))]
            for n, l in enumerate(same):
                for o, m in enumerate(labs):
                    if l[1] == m:
                        mx[n] = max(mx[n], recall[o][order[l[0]]])
            lab.append(same[np.argmax(np.array(mx))][0])
        else: lab.append(Counter(labs).most_common(1)[0][0])
    now_dict["labels"] = lab
    final_predictions.append(now_dict)

with open("predicted.json", "w") as f:
    json.dump(final_predictions, f, indent=4)

#### Analysis of how much each labels got ignored
with open('predicted.json','r') as f2:
    base = json.load(f2)

wrong = [0, 0, 0, 0, 0, 0]
for i in range(len(base)):
    for j in range(len(base[i]['labels'])):
        for k in range(len(seeds)):
            if base[i]['labels'][j] != all_predictions[k][i]['labels'][j]: wrong[k] += 1
print(wrong)
