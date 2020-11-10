# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:50:21 2020

@author: Kevin
"""

import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

path = "C:/Users/Kevin/Documents/GitHub/IntroToAICoursework" #relative or absolute path to data

#***********************Training/Validation Split***************************

filename_read = os.path.join(path, "vgsales.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
df = df.reindex(np.random.permutation(df.index)) # Usually a good idea to shuffle
mask = np.random.rand(len(df)) < 0.8
trainDF = pd.DataFrame(df[mask])
validationDF = pd.DataFrame(df[~mask])

print(f"Training DF: {len(trainDF)}")
print(f"Validation DF: {len(validationDF)}")

#***********************      K-Fold split       ***************************
kf = KFold(5)

fold = 1
for train_index, validate_index in kf.split(df):
    trainDF = pd.DataFrame(df.iloc[train_index, :])
    validateDF = pd.DataFrame(df.iloc[validate_index])
    print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    fold += 1