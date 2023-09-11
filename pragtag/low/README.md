### This subfolder explains the low condition.

Low condition is trained with [train_inputs_low.json](train_inputs_low.json) with four different classifiers. The details are presented in the table below. F1-score is computed on validation data. Third model is augmented with nlpaug package, which is [train_low_nlpaug.json](train_low_nlpaug.json).

| base model   | learning rate | seed | batch size | F1-score |
| ------------ | ------------- | ---- | ---------- | -------- |
| RoBERTa-base | 1e-5          | 42   | 8          | 0.7498   |
| RoBERTa-base | 2e-5          | 142  | 8          | 0.7667   |
| RoBERTa-base | 2e-5          | 242  | 8          | 0.7749   |
| BioBERT      | 1e-5          | 342  | 8          | 0.7534   |

F1000raw labels are in [seeds](seeds) directory. Using these data, majority labeling F1000raw dataset can be executed in [low.py](low.py). It will generate the labeled F1000raw dataset as [f1000_low_majority.json](f1000_low_majority.json).

We train a RoBERTa-base classifier with the aforementioned dataset and finally label the test data. [submission_low.zip](submission_low.zip) contains our best model's test data label.

We achieve `0.771` F1-mean score.
