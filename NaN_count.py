#!/usr/bin/env python
import numpy as np
import time
import pandas as pd
import localConfig as cfg

otherFeatures = ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l','BTagSF','ActualMu','LeptonSF', 'phiW', 'phiB1', 'phiB2', 'phiBB', 'phiJ3', 'phiL1']
trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']
trainFeatures.append("EventWeight")

nrows_signal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877
nrows_WlvZqq = 188395
nrows_WqqWlv = 334495

def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * mis_val / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(2)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n")
        #    "There are " + str(mis_val_table_ren_columns.shape[0]) +
        #      " columns that have missing values.")
        return mis_val_table_ren_columns

fraction = 1
start = time.time()
print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
tmp = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv',nrows = int(nrows_signal*fraction),usecols=trainFeatures,skipinitialspace=True)
print(missing_values_table(tmp[trainFeatures]))
del tmp

print("Reading -> 'stopWt_PwPy8_ade.csv'")
tmp = pd.read_csv('data/stopWt_PwPy8_ade.csv',nrows = int(nrows_stopWt*fraction),usecols=trainFeatures,skipinitialspace=True)
print(missing_values_table(tmp[trainFeatures]))
del tmp


print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
tmp = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',nrows = int((nrows_ttbar/10)*fraction),usecols=trainFeatures,skipinitialspace=True)
print(missing_values_table(tmp[trainFeatures]))
del tmp

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
tmp = pd.read_csv('data/WlvZqq_Sh221_ade.csv',nrows = int(nrows_WlvZqq*fraction),usecols=trainFeatures,skipinitialspace=True)
print(missing_values_table(tmp[trainFeatures]))
del tmp

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
tmp = pd.read_csv('data/WqqWlv_Sh221_ade.csv',nrows = int(nrows_WqqWlv*fraction),usecols=trainFeatures,skipinitialspace=True)
print(missing_values_table(tmp[trainFeatures]))
del tmp

print("Reading -> 'WJets_Sh221.csv'")
tmp = pd.read_csv('data/WJets_Sh221.csv',nrows = int(fraction*(nrows_Wjets/40)),usecols=trainFeatures,skipinitialspace=True)
print(missing_values_table(tmp[trainFeatures]))
del tmp
print "Reading time: ", (time.time() - start)
