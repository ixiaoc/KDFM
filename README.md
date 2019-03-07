# KDFM

This project includes a Tensorflow implementation of KDFM[1].The preprocessing contains three typlical stages:word embedding stage, bi_directional LSTM stage and tipical attention stage.

# Introduction

# Envirment

This code has been tested on Windows 10/64 bit, python environment 3.5.3.

# Usage
## Input Format
This implementation requires the input data in the following format:
- [ ] **Xi**: *[[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]*
    - *indi_j* is the feature index of feature field *j* of sample *i* in the dataset
- [ ] **Xv**: *[[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., {vali_1, vali_2, ..., vali_j, ...], ...]*
    - *vali_j* is the feature value of feature field *j* of sample *i* in the dataset
    - *vali_j* can be either binary (1/0, for binary/categorical features) or float (e.g., 4.1, for numerical features)
Please see `main.py` and `DataReader.py` an ecample how to prepare the data in required format for KDFM.
# Example
Folder `data` includes the data for the KDFM model.

To train KDFM model for this dataset, run

```
$ python main.py
```
Please see `main.py` and `DataReader.py` how to parse the raw dataset into the required format for KDFM.
You should tune the parameters for each model in order to get reasonable performance.

