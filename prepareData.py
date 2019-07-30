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

#myFeatures = [var for var in df_sinal.columns if var not in ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l']]
otherFeatures = ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l']
trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'phiW', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'phiB1', 'phiB2', 'phiBB', 'phiJ3', 'phiL1', 'BTagSF', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'ActualMu', 'LeptonSF', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']

nrows_sinal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877
nrows_WlvZqq = 188395
nrows_WqqWlv = 334495

start = time.time()
print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
df_sinal = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv')

print("Reading -> 'stopWt_PwPy8_ade.csv'")
df_stopWt = pd.read_csv('data/stopWt_PwPy8_ade.csv')

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
df_ttbar = pd.read_csv('data/ttbar_nonallhad_PwPy8_ade.csv',nrows=int(nrows_ttbar/10))

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
df_WlvZqq = pd.read_csv('data/WlvZqq_Sh221_ade.csv')

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
df_WqqWlv = pd.read_csv('data/WqqWlv_Sh221_ade.csv')

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv('data/WJets_Sh221.csv',nrows=int(nrows_Wjets/40))
print "Reading time: ", (time.time() - start)

data = None
for tmp in ["df_sinal","df_stopWt","df_ttbar","df_WlvZqq","df_WqqWlv","df_WJets"]:
        if df is None:
            data = pd.DataFrame(tmp)
        else:
            data = data.append(pandas.DataFrame(tmp), ignore_index=True)

del df_sinal, df_stopWt, df_ttbar, df_WlvZqq, df_WqqWlv, df_WJets

# Load the Data
print 'Datasets contain a total of', len(data), '(', data.weight.sum()*luminosity, 'weighted) events:'
