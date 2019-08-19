import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

## Input arguments. Pay speciall attention to the required ones.
parser = argparse.ArgumentParser(description='Process the command line options')
parser.add_argument('-f', '--fraction', type=float, default=0.3, help="The fraction of available data to be loaded")
args = parser.parse_args()
fraction = args.fraction

nrows_signal = 991141;
nrows_stopWt = 277816;
nrows_ttbar = 4168037;
nrows_Wjets = 16650877;
nrows_WlvZqq = 188395;
nrows_WqqWlv = 334495;

ttbar_fraction = 1.0/10.0
WJets_fraction = 1.0/40.0
chunksize = 1000

trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']
usecols = trainFeatures[:]
usecols.append("EventWeight")

lgbk = "/home/t3atlas/ev19u056/projetoWH/"
filepath = lgbk + "figures/"

def chunkReader(tmp):
    result = pd.DataFrame()
    for chunk in tmp:
        chunk.dropna(axis=0,how='any',subset=trainFeatures, inplace=True) # Dropping all rows with any NaN value
        result = result.append(chunk)
    return result

print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
tmp = pd.read_csv(lgbk+'data/qqWlvHbbJ_PwPy8MINLO_ade.csv',chunksize=chunksize,nrows = int(nrows_signal*fraction),usecols=usecols)
df_signal = chunkReader(tmp)

print("Reading -> 'stopWt_PwPy8_ade.csv'")
df_stopWt = pd.read_csv(lgbk+'data/stopWt_PwPy8_ade.csv',nrows = int(nrows_stopWt*fraction),usecols=usecols)

print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
tmp = pd.read_csv(lgbk+'data/ttbar_nonallhad_PwPy8_ade.csv',chunksize=chunksize,nrows = int((nrows_ttbar*ttbar_fraction)*fraction),usecols=usecols)
df_ttbar = chunkReader(tmp)

print("Reading -> 'WlvZqq_Sh221_ade.csv'")
df_WlvZqq = pd.read_csv(lgbk+'data/WlvZqq_Sh221_ade.csv',nrows = int(nrows_WlvZqq*fraction),usecols=usecols)

print("Reading -> 'WqqWlv_Sh221_ade.csv'")
df_WqqWlv = pd.read_csv(lgbk+'data/WqqWlv_Sh221_ade.csv',nrows = int(nrows_WqqWlv*fraction),usecols=usecols)

print("Reading -> 'WJets_Sh221.csv'")
df_WJets = pd.read_csv(lgbk+'data/WJets_Sh221.csv',nrows = int((nrows_Wjets*WJets_fraction)*(fraction)),usecols=usecols)

df_signal.EventWeight = df_signal.EventWeight/fraction
df_stopWt.EventWeight = df_stopWt.EventWeight/fraction
df_ttbar.EventWeight = df_ttbar.EventWeight/(fraction*ttbar_fraction)
df_WlvZqq.EventWeight = df_WlvZqq.EventWeight/fraction
df_WqqWlv.EventWeight = df_WqqWlv.EventWeight/fraction
df_WJets.EventWeight = df_WJets.EventWeight/(fraction*WJets_fraction)

i=1
nRow=2
nCol=2
k = 1
figure = plt.figure()
tmp = []
for var in trainFeatures:
    if (i == 5):
        plt.tight_layout()
        plt.savefig(filepath+str(k)+'_'+'_'.join(tmp)+'.pdf', bbox_inches='tight')
        plt.close()
        figure = plt.figure()
        del tmp [:];    tmp = []
        k += 1
        i = 1
    tmp.append(var)
    print k, var, " is plotting..."
    ax = figure.add_subplot(nRow, nCol,i)
    plt.hist(df_signal[var],weights=df_signal['EventWeight'],density=True,stacked=True,histtype='bar',label='signal')
    plt.hist(df_stopWt[var],weights=df_stopWt['EventWeight'],density=True,stacked=True,histtype='step',label='stopWt')
    plt.hist(df_ttbar[var],weights=df_ttbar['EventWeight'],density=True,stacked=True,histtype='step',label='ttbar')
    plt.hist(df_WlvZqq[var],weights=df_WlvZqq['EventWeight'],density=True,stacked=True,histtype='step',label='WlvZqq')
    plt.hist(df_WqqWlv[var],weights=df_WqqWlv['EventWeight'],density=True,stacked=True,histtype='step',label='WqqWlv')
    plt.hist(df_WJets[var],weights=df_WJets['EventWeight'],density=True,stacked=True,histtype='step',label='WJets')

    #plt.legend(loc='best')
    plt.grid(True)
    plt.title(var)
    i += 1
