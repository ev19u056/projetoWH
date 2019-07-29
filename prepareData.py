"""
Prepare Data for processing
"""

#!/usr/bin/env python

import root_numpy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import decomposition

import pandas as pd
import localConfig as cfg

df_sinal = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv', nrows=10)
trainvars = [var for var in df_sinal.columns if var not in ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l']]

print(len(trainvars))
print(trainvars)
