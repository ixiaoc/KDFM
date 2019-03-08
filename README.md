# KDFM

The project includes a Tensorflow implementation of KDFM [1].The preprocessing contains three typical stages: word embedding stage, bi_directional LSTM stage and topical attention stage.

# Introduction

We propose knowledge-based deep factorization machine (KDFM), a recommendation model inspired by techniques of factorization machine and attentive deep learning, and apply it to the recommendation service for mobile apps. The KDFM consists of two typical modules: a factorization machine for fine-grained low-order feature interactions and a deep neural network module for the deep extraction of hybrid types of features. By constructing the context-based topical preprocessing and handling the attention-based deep feature interactions, KDFM is able to make full use of the rich categorical and textual knowledge within app market, including reviews, descriptions, permissions, app name, download times, etc.

# Enviroment

Our source code has been tested on Windows 10/64 bit, python environment 3.5.3.

# Usage
## Input Format
The implementation requires the input data in the following format:
- [ ] **Xi**: *[[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]*
    - *indi_j* is the feature index of feature field *j* of sample *i* in the dataset
- [ ] **Xv**: *[[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., {vali_1, vali_2, ..., vali_j, ...], ...]*
    - *vali_j* is the feature value of feature field *j* of sample *i* in the dataset
    - *vali_j* can be either binary (1/0, for binary/categorical features) or float (e.g., 4.1, for numerical features)
- [ ] **y**: target of each sample in the dataset (1~5 for classification)

Please see `main.py` and `DataReader.py` an example how to prepare the data in required format for KDFM.
## Init and train a model
```python
import tensorflow as tf
from metrics import gini_norm, mse_norm, mse
from KDFM import KDFM

# params
kdfm_params = {
    "use_afm": True,
    "use_deep": True,
    "feature_size_one_hot": 1,
    "field_size_one_hot": 3,
    "feature_size_multi_value": 0,
    "field_size_multi_value": 0,
    "embedding_size": 8,
    "attention_size": 10,

    "deep_layers": [32, 32, 32],
    "dropout_deep": [0.5, 0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,

    "epoch": 30,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "adam",

    "random_seed": config.RANDOM_SEED,
    "l2_reg": 0.1,

    "rnn_size": 100,
    "num_rnn_layers": 1,
    "keep_lstm": 0.5,
    "num_unroll_steps": 100,  # 句子长度
    "verbose": True,
    "topics": 1
}

# prepare training and validation data in the required format
    Xi_train, Xv_train, y_train = data_parser.parse(...)
    Xt_train, Xm_train = read_text_data(...)
    Xi_test, Xv_test, y_test = data_parser.parse(...)
    Xt_test, Xm_test = read_text_data(...)

# init a KDFM model
kdfm = KDFM(**kdfm_params)

# fit a KDFM model
kdfm.fit(Xi_train, Xv_train, Xim_train, Xvm_train, Xt_train, y_train)

# make prediction
kdfm.predict(Xi_valid, Xv_valid, Xim_valid, Xvm_vaild, Xt_valid)

# evaluate a trained model
kdfm.evaluate(XXi_valid, Xv_valid, Xim_valid, Xvm_vaild, Xt_valid, y_valid)
```
# Example
Folder `data` includes the data for the KDFM model.

To train KDFM model for this dataset, run

```
$ python main.py
```
Please see `main.py` and `DataReader.py` how to parse the raw dataset into the required format for KDFM.
You should tune the parameters for each model in order to get reasonable performance.

