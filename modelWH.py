import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

'''
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
'''
import time

nrows_sinal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877
nrows_WlvZqq = 188395
nrows_WqqWlv = 334495

nrowsWjets = 0.1*nrows_Wjets
print("Tentativa de importar 10% dos dados")
nrowsStopWt = int((nrows_stopWt/nrows_Wjets)*nrowsWjets)
nrowsSinal = int((nrows_sinal/nrows_Wjets)*nrowsWjets)
nrowsTtbar = int((nrows_ttbar/nrows_Wjets)*nrowsWjets)
nrowsWlvZqq = int((nrows_WlvZqq/nrows_Wjets)*nrowsWjets)
nrowsWqqWlv = int((nrows_WqqWlv/nrows_Wjets)*nrowsWjets)

start = time.time()
print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
df_sinal = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv', nrows=nrowsSinal)

print("Reading -> 'stopWt_PwPy8_ade.csv'")
df_stopWt = pd.read_csv('data/stopWt_PwPy8_ade.csv',nrows=nrowsStopWt)

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
df_ttbar = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',nrows=nrowsTtbar)

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
df_WlvZqq = pd.read_csv('data/WlvZqq_Sh221_ade.csv',nrows=nrowsWlvZqq)

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
df_WqqWlv = pd.read_csv('data/WqqWlv_Sh221_ade.csv',nrows=nrowsWqqWlv)
#print("'WqqWlv_Sh221_ade.csv' -> reading over")

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv('data/WJets_Sh221.csv',nrows=nrowsWjets)
print("'WJets_Sh221.csv' has been read")

end = time.time()
print("Reading time: ", (end - start), "s")

trainvars=[var for var in df_sinal.columns if var not in ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavour', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'EventFlavor','eventFlagMerged/l','eventFlagResolved/l']]

for var in trainvars:
    print("Plotting: " + var)
    plt.figure()
    plt.hist(df_WJets[var],density=True,stacked=True,histtype='bar',label='WJets')
    plt.hist(df_ttbar[var],density=True,stacked=True,histtype='bar',label='ttbar')
    plt.hist(df_sinal[var],density=True,stacked=True,histtype='bar',label='signal')
    plt.hist(df_stopWt[var],density=True,stacked=True,histtype='bar',label='stopWt')
    plt.hist(df_WlvZqq[var],density=True,stacked=True,histtype='bar',label='WlvZqq')
    plt.hist(df_WqqWlv[var],density=True,stacked=True,histtype='bar',label='WqqWlv')
    plt.legend(prop={'size': 10})

    plt.grid(True)
    plt.title(var)
    plt.savefig('figures/'+var+'.png')
    plt.clf()
