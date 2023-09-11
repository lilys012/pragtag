### This subfolder explains datasets used among all conditions.

[train_inputs.json](train_inputs.json), [test_inputs.json](test_inputs.json), and [secret_test_inputs.json](secret_test_inputs.json) are all given from the shared task organizers. Please rename respective dataset into above.

We randomly sample 18 reviews from [train_inputs.json](train_inputs.json), excluding those in [train_inputs_low.json](../low/train_inputs_low.json), to use as validation set. They can be found in [val_inputs.json](val_inputs.json) and [val_labels.json](val_labels.json). We leave ids of validation set in [val_ids.json](val_ids.json) for reproducibility.

Auxiliary datasets should also be acquired. `F1000raw` and `ARR-22` dataset are segmented into sentences using `nltk` package, which should be named as [f1000_train_inputs_nonlabeled.json](f1000_train_inputs_nonlabeled.json) and [arr22_train_inputs.json](arr22_train_inputs.json). Unlike `F1000raw`, `ARR-22` contains sections of review, which are also included as labels. [aux_extract.py](aux_extract.py) contains the above process.

Please use your respective `DATANAME` and `rootdir` in the code.
