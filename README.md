## Enhancing Cross-Domain Generalization through Data Augmentation with Reduced Uncertainty

This is a repository containing the code of our submission to ArgMining @ EMNLP 2023.

The task is constructed in three conditions: [full](pragtag/full), [low](pragtag/low), and [zero](pragtag/zero). Each subdirectory has its own README.md which instructs how to conduct experiments. Visit [pragtag](pragtag) folder for further details.

There are some data that are universally shared across all conditions, which are stored in [public_dat](pragtag/public_dat). Please check `Data` section below for specific instructions.

Besides data augmentation strategies, finetuning, predicting, and evaluating are performed at seperate directories.

### Finetune

[finetune_baseline.py](baseline/finetune_baseline.py) is a file of finetuning an encoder classifier. Just run the below.

```bash
python3 finetune_baseline.py <train data path> <output path> <model name>
```

Various hyperparameters are subject to change. We commented all hyperparamters needed for each experiment, so please feel free to adjust the below hyperparameters in the file.

`seed, num_training_epochs, learning_rate, per_device_train_batch_size, callbacks`

Please be aware that on zero condition, only four labels are classified. The following has to be changed.

1. LABELS at [load.py](evaluation/load.py).
2. CLASS_MAP and iCLASS_MAP at [utils.py](baseline/utils.py)

### Prediction

[predict_baseline.py](baseline/predict_baseline.py) allows trained model to predict labels when data is given. Just run the below.

```bash
python3 predict_baseline.py <test input data path> <model checkpoint path> <output path>
```

### Evaluation

[main.py](evaluation/main.py) will evaluate the F1-scores across all domains.

```bash
python3 main.py <input_path> <output_path>

```

Here, the input path should point to a directory containing a folder "ref" with the true labels (or training data with labels) under the name `test_labels.json`, and a folder res with the predicted labels (under `predicted.json`).

### Data

Please rename the dataset and place it where described in the respective README.md.

Below are links to obtain the dataset.

#### Public Data

Gain access to the data from [workshop page](https://codalab.lisn.upsaclay.fr/competitions/13334#learn_the_details) and rename them accordingly.

#### Auxiliary Data

Click on the link provided in the shared task and request the data. After confirmation (requires prior registration with the shared task), you will receive the auxiliary data. For conveniently loading it, checkout the associated [github repo](https://github.com/UKPLab/nlpeer).
