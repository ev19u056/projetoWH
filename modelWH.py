import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import ForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input, Embedding, Reshape, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, History

nrows_sinal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877

nrowsWjets = 16650877
nrowsStopWt = int((nrows_stopWt/nrows_Wjets)*nrowsWjets)
nrowsSinal = int((nrows_sinal/nrows_Wjets)*nrowsWjets)
nrowsTtbar = int((nrows_ttbar/nrows_Wjets)*nrowsWjets)

print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
df_sinal = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv', nrows=nrowsSinal)

print("Reading -> 'stopWt_PwPy8_ade.csv'")
df_stopWt = pd.read_csv('data/stopWt_PwPy8_ade.csv',nrows=nrowsStopWt)

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
df_ttbar = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',nrows=nrowsTtbar)

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv('data/WJets_Sh221.csv',nrows=nrowsWjets)

trainvars=[var for var in df_sinal.columns if var not in ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavour', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'EventFlavor','eventFlagMerged/l','eventFlagResolved/l']]

for var in trainvars:

    print("Plotting: " + var)
    plt.figure()
    plt.hist(df_WJets[var],density=True,stacked=True,histtype='bar',label='WJets')
    plt.hist(df_ttbar[var],density=True,stacked=True,histtype='bar',label='ttbar')
    plt.hist(df_sinal[var],density=True,stacked=True,histtype='bar',label='sinal')
    plt.hist(df_stopWt[var],density=True,stacked=True,histtype='bar',label='stopWt')
    plt.legend(prop={'size': 10})

    plt.grid(True)
    plt.title(var)
    plt.savefig('figures/'+var+'.png')
    plt.clf()
    #    if i==5:
    #       break
    #    i += 1