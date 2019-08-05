"""
Prepare Data for processing
"""

#!/usr/bin/env python
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import decomposition

import pandas as pd
import localConfig as cfg

#preselection =
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#myFeatures = [var for var in df_sinal.columns if var not in ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l']]
otherFeatures = ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l','BTagSF','ActualMu','LeptonSF', 'phiW', 'phiB1', 'phiB2', 'phiBB', 'phiJ3', 'phiL1']
trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']
usecols = trainFeatures[:]; usecols.append("EventWeight")
scalingFeatures = ['mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']

nrows_signal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877
nrows_WlvZqq = 188395
nrows_WqqWlv = 334495

ttbar_fraction = 1.0/10.0
WJets_fraction = 1.0/40.0

def chunkReader(tmp):
    result = pd.DataFrame()
    for chunk in tmp:
        chunk.dropna(axis=0,how='any',subset=trainFeatures, inplace=True) # Dropping all rows with any NaN value
        result = result.append(chunk)
    #del tmp, chunk
    return result

fraction = 0.05
chunksize = 1000
start = time.time()
print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
tmp = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv',chunksize=chunksize,nrows = int(nrows_signal*fraction),usecols=usecols)
df_signal = chunkReader(tmp)
df_signal[["EventWeight"]] = df_signal[["EventWeight"]]/fraction

print("Reading -> 'stopWt_PwPy8_ade.csv'")
df_stopWt = pd.read_csv('data/stopWt_PwPy8_ade.csv',nrows = int(nrows_stopWt*fraction),usecols=usecols)
df_stopWt[["EventWeight"]] = df_stopWt[["EventWeight"]]/fraction

print((nrows_ttbar*ttbar_fraction)*fraction)

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
tmp = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',chunksize=chunksize,nrows = int((nrows_ttbar*ttbar_fraction)*fraction),usecols=usecols)
df_ttbar = chunkReader(tmp)
df_ttbar[["EventWeight"]] = df_ttbar[["EventWeight"]]/(fraction*ttbar_fraction)

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
df_WlvZqq = pd.read_csv('data/WlvZqq_Sh221_ade.csv',nrows = int(nrows_WlvZqq*fraction),usecols=usecols)
df_WlvZqq[["EventWeight"]] = df_WlvZqq[["EventWeight"]]/fraction

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
df_WqqWlv = pd.read_csv('data/WqqWlv_Sh221_ade.csv',nrows = int(nrows_WqqWlv*fraction),usecols=usecols)
df_WqqWlv[["EventWeight"]] = df_WqqWlv[["EventWeight"]]/fraction

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv('data/WJets_Sh221.csv',nrows = int((nrows_Wjets*WJets_fraction)*(fraction)),usecols=usecols)

#df_WJets[["EventWeight"]] = df_WJets[["EventWeight"]]*(fraction/40)
df_WJets[["EventWeight"]] = df_WJets[["EventWeight"]]/(fraction*WJets_fraction)
print "Reading time: ", (time.time() - start)

df_signal["category"] = 1
df_stopWt["category"] = 0
df_ttbar["category"] = 0
df_WlvZqq["category"] = 0
df_WqqWlv["category"] = 0
df_WJets["category"] = 0

print(df_ttbar.category)

data = None
for tmp in [df_signal,df_stopWt,df_ttbar,df_WlvZqq,df_WqqWlv,df_WJets]:
        if data is None:
            data = tmp
            del tmp
        else:
            data.append(tmp, ignore_index=True)
            del tmp

print(df_ttbar.category)
del df_stopWt, df_ttbar, df_WlvZqq, df_WqqWlv, df_WJets, df_signal
print 'Datasets contain a total of', len(data)#, '(', data.EventWeight.sum(), 'weighted) events:'


'''
data = data.sample(frac=1,random_state=seed).reset_index(drop=True)

Dev = 0.8
Dev_len = int(len(data)*Dev)
Val_len = len(data)-Dev_len

print 'Datasets contain a total of', len(data)#, '(', data.EventWeight.sum(), 'weighted) events:'

XDev = data[trainFeatures].ix[0:Dev_len-1,:]
YDev = data[["category"]].ix[0:Dev_len-1,:]
XVal = data[trainFeatures].ix[Dev_len:,:]
YVal = data[["category"]].ix[Dev_len:,:]
'''
Dev, Val, Test = np.split(data.sample(frac=1,random_state=seed).reset_index(drop=True), [int(0.8*len(data)), int(0.9*len(data))])
del data

XDev = Dev[trainFeatures]
YDev = Dev[["category"]]
weightDev = np.ravel(Dev.EventWeight)
del Dev

XVal = Val[trainFeatures]
YVal = Val[["category"]]
weightVal = np.ravel(Val.EventWeight)
del Val

XTest = Test[trainFeatures]
YTest = Test[["category"]]
weightTest = np.ravel(Test.EventWeight)
del Test

print 'XDev: ', len(XDev), ' YDev: ', len(YDev), ' weightDev: ', len(weightDev)
print 'XVal: ', len(XVal), ' YVal: ', len(YVal), ' weightVal: ', len(weightVal)
print 'XTest: ', len(XTest), ' YTest: ', len(YTest), ' weightTest: ', len(weightTest)

print(YDev)
# print "Fitting the scaler and scaling the input variables ..."
# scaler = StandardScaler().fit(XDev[scalingFeatures])
# XDev[scalingFeatures] = scaler.transform(XDev[scalingFeatures])
# XVal[scalingFeatures] = scaler.transform(XVal[scalingFeatures])
#scalerfile = 'scaler_'+train_DM+'.sav'
#joblib.dump(scaler, scalerfile)
