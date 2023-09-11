import json
import random

random.seed(42)

'''
######################
Source code for transforming labels into Structure and Other
######################
'''

#### reference auxiliary data : ARR dataset
with open("../public_dat/arr22_train_inputs.json", "r") as f1:
    aux = json.load(f1)

#### 4-class predicted labels
with open("4_class_predicted.json", "r") as f2:
    pred = json.load(f2)

ans = []
for t, p in zip(aux, pred):
    labels = []
    for a, b in zip(t["sentences"], p["labels"]):
        #### if sentence ends with ':' or is equal or less than 5 words, label as Structure
        if a.strip()[-1] == ':' or len(a.strip().split())<=5:
            labels.append('Structure')
        #### randomly label 15% of Weakness and Recap to Other
        elif (b == "Weakness" or b == "Recap") and random.random() <= 0.15:
            labels.append('Other')
        else: labels.append(b)
    ans.append({'id':t['id'], 'labels':labels})

#### export labeled test data to standard evaluation format
with open("predicted.json", "w") as f3:
    json.dump(ans, f3, indent=4)