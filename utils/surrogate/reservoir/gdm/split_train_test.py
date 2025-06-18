import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import os
from pathlib import Path

def split_train_test(filename, train_name, test_name, split_ratio=0.7, delimiter='\t'):
    #poly = mpg.MultPolygon()
    #poly.read_history(filename, delimiter=delimiter)

    df = pd.read_csv(filename, delimiter=delimiter)

    data = {}
    variants = []

    for var_id, var_df in df.groupby('variant_id'):
        data[var_id] = var_df
        variants.append(var_id)

    variants = np.array(variants)
    np.random.shuffle(variants)

    split = int(np.floor(variants.shape[0] * split_ratio))

    df_train = []
    df_test = []

    for var_id in variants[:split]:
        df_train.append(data[var_id])
    for var_id in variants[split:]:
        df_test.append(data[var_id])

    df_train = pd.concat(df_train)
    df_train.to_csv(train_name, sep=delimiter)

    df_test = pd.concat(df_test)
    df_test.to_csv(test_name, sep=delimiter)

#split_train_test('data/dataset5/train.txt', 'data/dataset5/train0.txt', 'data/dataset5/test0.txt', 0.7)
#split_train_test('data/dataset5/test0.txt', 'data/dataset5/test0.txt', 'data/dataset5/test1.txt', 0.6)
    
#split_train_test('data/dataset8/dataset_v8_test_dif_var_old_format.csv', 'data/dataset8/train0.txt', 'data/dataset8/test0.txt', 0.7)
#split_train_test('data/dataset8/test0.txt', 'data/dataset8/test0.txt', 'data/dataset8/test1.txt', 0.6)
    
split_train_test('data/dataset24/dataset.csv', 'data/dataset24/train0.txt', 'data/dataset24/test0.txt', 0.7, delimiter=';')

#split_train_test('data/dataset_v2/Input_dataset_v2_cutted_before_1004_date.csv', 'data/dataset_v2/train0.txt', 'data/dataset_v2/test0.txt', 0.7, delimiter=';')