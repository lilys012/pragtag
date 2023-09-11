import sys
import random

import numpy as np
import torch

from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback, IntervalStrategy

from utils import CLASS_MAP, to_context_free_format, iCLASS_MAP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("TRAINING on ", device)

seed = 242
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# training data
train_data_path = sys.argv[1]


def preprocess(item):
    item["label"] = torch.tensor(CLASS_MAP[item["label"]]).unsqueeze(0)

    return item


model_path = sys.argv[3]

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize(examples):
    toks = tokenizer.batch_encode_plus(examples["txt"], padding="max_length", max_length=512, truncation=True,
                                       return_tensors="pt")
    toks["labels"] = examples["label"]

    return toks


full_data = Dataset.from_list(to_context_free_format(train_data_path)) \
    .map(preprocess) \
    .shuffle(seed=seed) \
    .train_test_split(test_size=0.1) \
    .map(tokenize, batched=True)

# model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(iCLASS_MAP)).to(device)

# fine-tuning
training_args = TrainingArguments(
    output_dir=sys.argv[2] + '/results',
    num_train_epochs=25,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=sys.argv[2] + '/logs',
    logging_steps=5000,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    load_best_model_at_end=True,
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_data["train"],
    eval_dataset=full_data["test"],
    tokenizer=tokenizer,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]
)

# actual training
trainer.train()

# loading for prediction
best_path = sys.argv[2] + '/results/best_roberta'
trainer.save_model(best_path)

print("Done")
