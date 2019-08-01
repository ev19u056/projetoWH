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
otherFeatures = ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l','BTagSF','ActualMu','LeptonSF']
trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'phiW', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'phiB1', 'phiB2', 'phiBB', 'phiJ3', 'phiL1', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']

'''
start = time.time()
print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
df_signal = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv')

print("Reading -> 'stopWt_PwPy8_ade.csv'")
df_stopWt = pd.read_csv('data/stopWt_PwPy8_ade.csv')

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
df_ttbar = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',nrows=int(nrows_ttbar/20))

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
df_WlvZqq = pd.read_csv('data/WlvZqq_Sh221_ade.csv')

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
df_WqqWlv = pd.read_csv('data/WqqWlv_Sh221_ade.csv')

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv('data/WJets_Sh221.csv',nrows=int(nrows_Wjets/50))
print "Reading time: ", (time.time() - start)
'''

nrows_signal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877
nrows_WlvZqq = 188395
nrows_WqqWlv = 334495

def chunkReader(tmp):
    result = pd.DataFrame()
    for chunk in tmp:
        chunk.dropna(axis=0,how='any',subset=trainFeatures, inplace=True) # Dropping all rows with any NaN value
        result = result.append(chunk)
    del tmp, chunk
    return result

i = 0.1
chunksize = 1000
start = time.time()
print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
tmp = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv',chunksize=chunksize,nrows = int(nrows_signal*i))
df_signal = chunkReader(tmp)
del tmp

print("Reading -> 'stopWt_PwPy8_ade.csv'")
tmp = pd.read_csv('data/stopWt_PwPy8_ade.csv',chunksize=chunksize,nrows = int(nrows_stopWt*i))
df_stopWt = chunkReader(tmp)
del tmp

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
df_ttbar = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',nrows = int(nrows_ttbar*i))

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
df_WlvZqq = pd.read_csv('data/WlvZqq_Sh221_ade.csv',nrows = int(nrows_WlvZqq*i))

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
df_WqqWlv = pd.read_csv('data/WqqWlv_Sh221_ade.csv',nrows = int(nrows_WqqWlv*i))

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv('data/WJets_Sh221.csv',nrows = int(nrows_Wjets/40))
print "Reading time: ", (time.time() - start)

df_signal["category"] = 1
df_stopWt["category"] = 0
df_ttbar["category"] = 0
df_WlvZqq["category"] = 0
df_WqqWlv["category"] = 0
df_WJets["category"] = 0

data = None
for tmp in [df_signal,df_stopWt,df_ttbar,df_WlvZqq,df_WqqWlv,df_WJets]:
        if data is None:
            data = tmp
            del tmp
        else:
            data = data.append(tmp, ignore_index=True)
            del tmp

del df_stopWt, df_ttbar, df_WlvZqq, df_WqqWlv, df_WJets, df_signal
data = data.sample(frac=1,random_state=seed).reset_index(drop=True)
# Load the Data

Dev = 0.75
Dev_len = int(len(data)*Dev)
Val_len = len(data)-Dev_len

print 'Datasets contain a total of', len(data)#, '(', data.EventWeight.sum(), 'weighted) events:'
XDev = data[trainFeatures].ix[0:Dev_len-1,:]
YDev = data[["category"]].ix[0:Dev_len-1,:]
XVal = data[trainFeatures].ix[Dev_len:,:]
YVal = data[["category"]].ix[Dev_len:,:]

del data
print 'XDev: ', len(XDev), ' YDev: ', len(YDev)
print 'XVal: ', len(XVal), ' YVal: ', len(YVal)

print "Fitting the scaler and scaling the input variables ..."
scaler = StandardScaler().fit(XDev)
XDev = scaler.transform(XDev)
XVal = scaler.transform(XVal)
#scalerfile = 'scaler_'+train_DM+'.sav'
#joblib.dump(scaler, scalerfile)
