import json
import torch
from collections import Counter

'''
######################
Source code for majority labeling F1000raw dataset
######################
'''

CLASS_MAP = {
    cn: i for i, cn in enumerate(["Strength", "Weakness", "Todo", "Structure", "Recap", "Other"])
}

savepath = "./"
datapath = "../public_dat/f1000_train_inputs_nonlabeled.json"

#### Read the predictions
all_predictions = []
seeds = ['seeds/seed242.json', 'seeds/seed42.json', 'seeds/seed142.json', 'seeds/seed342.json']
for address in seeds:
    with open(address, "r") as t1:
        temp = json.load(t1)
        all_predictions.append(temp)

with open("train_inputs_low.json", "r") as f0:
    train = json.load(f0)
with open("../public_dat/f1000_train_inputs_nonlabeled.json", "r") as f1:
    f1000 = json.load(f1)

#### Labeling starts
majorcnt = 0
final_predictions = []
for i in range(len(all_predictions[0])):
    now_dict = {"id": all_predictions[0][i]["id"]}
    lab = []
    for j in range(len(all_predictions[0][i]["labels"])):
        labs = [all_predictions[k][i]["labels"][j] for k in range(len(seeds))]
        #### Majority label
        lab.append(Counter(labs).most_common(1)[0][0])
        majorcnt += 1
    now_dict["labels"] = lab
    final_predictions.append(now_dict)

f1000_dict = {}
for fs in f1000:
    f1000_dict[fs["id"]] = fs

#### Reconstruct data into acceptible format
reslist = []
for a in final_predictions:
    ref = f1000_dict[a["id"]]
    temp = ref
    temp["labels"] = a["labels"]
    reslist.append(temp)
reslist.extend(train)
print(len(reslist), len(train), len(f1000))

with open(savepath + "/f1000_low_majority.json", "w+") as f:
    json.dump(reslist, f, indent=4)

#### The following code is for Consensus labeling

# allcnt = 0
# final_predictions = []
# for i in range(len(all_predictions[0])):
#     now_dict = {"id": all_predictions[0][i]["id"]}
#     lab = []
#     for j in range(len(all_predictions[0][i]["labels"])):
#         labs = [all_predictions[k][i]["labels"][j] for k in range(len(seeds))]
#         if Counter(labs).most_common(1)[0][1] == len(seeds):
#             lab.append(Counter(labs).most_common(1)[0][0])
#             allcnt += 1
#         else:
#             lab.append('None')
#     now_dict["labels"] = lab
#     final_predictions.append(now_dict)

# with open(savepath + "/f1000_low_consensus.json", "w+") as f:
#     json.dump(final_predictions, f, indent=4)

# print("Done")
# print(majorcnt, allcnt)
