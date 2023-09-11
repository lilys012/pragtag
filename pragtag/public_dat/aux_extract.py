import os
import json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

DATANAME = "F1000-22" # or "ARR-22"
rootdir = f"{DATANAME}/data"

# id, sentences, labels, pid, domain

retlist = []
cnt = 0
if DATANAME == "ARR-22":
    reviewidx = ["paper_summary", "summary_of_strengths", "summary_of_weaknesses", "comments,_suggestions_and_typos"]
    reviewlabel = ["Recap", "Strength", "Weakness", "Todo"]
    with os.scandir(rootdir) as entries:
            for entry in tqdm(entries):
                if entry.is_dir():
                    cnt += 1
                    reviews = []
                    
                    localdir = rootdir+"/"+entry.name+"/v1/reviews.json"
                    with open(localdir, "r") as f1:
                        reviews = json.load(f1)
                        
                    for review in reviews:
                        retdict = {"id":review["rid"]}
                        paragraph = []
                        labels = []
                        for i, ridx in enumerate(reviewidx):
                            paragraph.append(review["report"][ridx])
                            labels.append(reviewlabel[i])
                        retdict["sentences"] = paragraph
                        retdict["labels"] = labels
                        retdict["pid"] = entry.name
                        retlist.append(retdict)

    with open("arr22_train_inputs.json", "w") as f2:
        json.dump(retlist, f2, indent=4)
    print(cnt)

elif DATANAME == "F1000-22":
    empty = 0
    with os.scandir(rootdir) as entries:
        for entry in tqdm(entries):
            if entry.is_dir():
                cnt += 1
                reviews = []
                    
                localdir = rootdir+"/"+entry.name+"/v1/reviews.json"
                try:
                    with open(localdir, "r") as f1:
                        reviews = json.load(f1)
                except:
                    pass
                    
                for review in reviews:
                    if not review["report"]["main"] or review["report"]["main"] == "":
                        empty += 1
                        continue
                    retdict = {"id":review["rid"]}
                    retdict["sentences"] = sent_tokenize(review["report"]["main"])
                    retdict["pid"] = entry.name
                    retlist.append(retdict)

    with open("f1000_train_inputs_nonlabeled.json", "w") as f2:
        json.dump(retlist, f2, indent=4)
    print(cnt)
    print(empty)


