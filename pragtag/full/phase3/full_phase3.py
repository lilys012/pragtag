import json
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASS_MAP = {
    cn: i for i, cn in enumerate(["Strength", "Weakness", "Todo", "Structure", "Recap", "Other"])
}
iCLASS_MAP = {
    v: k for k, v in CLASS_MAP.items()
}

'''
######################
Source code for recall labeling F1000raw dataset
######################
'''

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

with open("../public_dat/f1000_train_inputs_nonlabeled.json", "r") as f1:
    f1000 = json.load(f1)

#### Initially all labels are None
for i in range(len(f1000)):
    f1000[i]['labels'] = ['None' for _ in range(len(f1000[i]['sentences']))]

#### Order of descending recall with respective auxiliary data
order = ['Structure', 'Todo', 'Strength', 'Recap', 'Weakness']
names = ['tags/final/predicted_f1000.json', 'tags/predicted_nlpaug_50.json', 'tags/predicted_nlpaug_2567.json', 'tags/predicted_nlpaug_3020.json', 'tags/predicted_nlpaug_30.json']

for label, name in zip(order, names):
    with open(name, "r") as f2:
        candidates = json.load(f2)

    for i in range(len(f1000)):
        assert f1000[i]['id'] == candidates[i]['id']
        #### Recall label
        for j in range(len(f1000[i]['labels'])):
            if f1000[i]['labels'][j] != 'None': continue
            elif candidates[i]['labels'][j] == label: f1000[i]['labels'][j] = label

#### Label remainders into Other
last = 'Other'
for i in range(len(f1000)):
    for j in range(len(f1000[i]['labels'])):
        if f1000[i]['labels'][j] == 'None': f1000[i]['labels'][j] = 'Other'

#### Extend train_inputs
with open("train_inputs_full.json", "r") as f10:
    train = json.load(f10)
f1000.extend(train)

#### Save data
with open("f1000_recall_labeled.json", "w") as f11:
    json.dump(f1000, f11, indent=4)