import json
import torch
import sys
from collections import Counter
import random

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter

CLASS_MAP = {
    cn: i for i, cn in enumerate(["Strength", "Weakness", "Todo", "Structure", "Recap", "Other"])
}
iCLASS_MAP = {
    v: k for k, v in CLASS_MAP.items()
}

# converts the sequence of reviews with sentences to just the sentences. The "sid" field is composed of the review id and the sentence number to be reversable
def to_context_free_format(path):
    with open(path, "r") as f:
        d = json.load(f)

    res = []
    for i in d:
        for c, s in enumerate(i["sentences"]):
            k = {"sid": i["id"] + "@" + str(c), "txt": s}
            if "labels" in i:
                k["label"] = i["labels"][c]

            res += [k]

    return res

# converts the given predictions on sentence level back to review-level predictions
def predictions_to_evaluation_format(pred):
    res = {}
    for i in pred:
        iid, six = tuple(i["sid"].split("@"))
        res[iid] = res.get(iid, {})
        res[iid][int(six)] = i["label"]

    return [{"id": k, "labels": [v[j] for j in range(len(v))]} for k, v in res.items()]

'''
######################
Source code for majority labeling F1000raw dataset
######################
'''

savepath = "seeds"
datapath = "../public_dat/f1000_train_inputs_nonlabeled.json"
model_path = sys.argv[1]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("EVALUATING on ", device)

def preprocess(item):
    if "label" in item:
        item["label"] = torch.tensor(CLASS_MAP[item["label"]]).unsqueeze(0)

    return item

def tokenize(examples):
    toks = best_tok.batch_encode_plus(examples["txt"], padding="max_length", max_length=512, truncation=True, return_tensors="pt")

    return toks

#### Labeling auxiliary data with each model
all_predictions = []
seeds = [42, 142, 242, 342, 442]
for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    best_model_path = model_path + f"seed{seed}/results/best_roberta"
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    best_model.eval()
    best_model.to(device)
    best_tok = AutoTokenizer.from_pretrained(best_model_path)


    eval_data_path = datapath
    eval_data = Dataset.from_list(to_context_free_format(eval_data_path))\
        .map(preprocess)\
        .map(tokenize, batched=True)

    batch_size = 1
    inputs = DataLoader(eval_data, batch_size=batch_size)

    predictions = []
    for batch_inputs in tqdm(inputs, desc="Iterating over input"):
        batch_outputs = best_model(input_ids = torch.stack(batch_inputs["input_ids"]).transpose(1,0).to(device),
                            attention_mask=torch.stack(batch_inputs["attention_mask"]).transpose(1,0).to(device))
        batch_predictions = torch.argmax(batch_outputs.logits, dim=1)

        predictions += [{"sid": batch_inputs["sid"][i], "label": iCLASS_MAP[batch_predictions[i].item()]} for i in range(batch_size)]
    r = predictions_to_evaluation_format(predictions)
    all_predictions.append(r)
    with open(savepath + f"/predicted_{seed}.json", "w+") as f:
        json.dump(r, f, indent=4)

#### Majority labeling
majorcnt = 0
final_predictions = []
for i in range(len(all_predictions[0])):
    now_dict = {"id": all_predictions[0][i]["id"]}
    lab = []
    for j in range(len(all_predictions[0][i]["labels"])):
        labs = [all_predictions[k][i]["labels"][j] for k in range(len(seeds))]
        lab.append(Counter(labs).most_common(1)[0][0])
        majorcnt += 1
    now_dict["labels"] = lab
    final_predictions.append(now_dict)

with open("f1000_full_majority.json", "w+") as f:
    json.dump(final_predictions, f, indent=4)

#### Consensus labeling

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

# with open(savepath + "/predicted_all.json", "w+") as f:
#     json.dump(final_predictions, f, indent=4)

# print("Done")
# print(majorcnt, allcnt)

#### concatenating with full data
with open("../public_dat/f1000_train_inputs_nonlabeled.json", "r") as f10:
    f1000 = json.load(f10)

with open("f1000_full_majority.json", "r") as f11:
    labels = json.load(f11)

res = []
for a, b in zip(f1000, labels):
    assert a["id"] == b["id"]
    sents = []
    labs = []
    for i in range(len(b["labels"])):
        if b["labels"][i] != "None":
            sents.append(a["sentences"][i])
            labs.append(b["labels"][i])
    if len(labs):
        res.append({"id":a["id"], "sentences":sents, "pid":a["pid"], "labels":labs})

random.shuffle(res)

with open("train_inputs_full.json", "r") as f3:
    full = json.load(f3)
res.extend(full)
print(len(res))

with open("f1000_majority_train_inputs_full.json", "w") as f4:
    json.dump(res, f4, indent=4)