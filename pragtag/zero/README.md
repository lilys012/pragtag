### This subfolder explains the zero condition.

Zero condition uses [ARR dataset](../public_dat/aux_train_inputs.json) for initial classifier. This model lables the test data into one of the four labels: Recap, Strength, Weakness, and Todo.

Please provide the previously mentioned labels as `4_class_predicted.json` and run [zero.py](zero.py). It will automatically generate `predicted.json`. Then, run the evaluation code.

[submission_zero.zip](submission_zero.zip) contains our best model's test data label. [submission_final_zero.zip](submission_final_zero.zip) contains our best model's test data label, including the secret domain.

We also explore diverse segmentation of ARR dataset. 

[arr22_train_total_inputs.json](arr22_train_total_inputs.json) doesn't segment review section into sentences, but uses the whole paragraph. 

[arr22_train_nlpaug_threshold_cl.json](arr22_train_nlpaug_threshold_cl.json) employs the synonym generaton technique, apply BERTSCORE threshold, and remove single-word sentneces. Code for this process can be found in [aux_augment.ipynb](aux_augment.ipynb) Both didn't outperform the previously constructed [ARR dataset](../public_dat/arr22_train_inputs.json).

We achieve `0.516` F1-mean score and `0.517` including secret domain.
