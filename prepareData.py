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
myFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'phiW', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'phiB1', 'phiB2', 'phiBB', 'phiJ3', 'phiL1', 'BTagSF', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'ActualMu', 'LeptonSF', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']
inputBranches = list(myFeatures)

number_of_events_print = 1 # ???

print "Loading datasets..."
dataDev, dataVal = StopDataLoader(cfg.loc, inputBranches, selection=preselection,
                    suffix=suffix, signal=train_DM, test=test_point,
                    fraction=fraction, useSF=True)

df_sinal = pd.read_csv('data/qqWlvHbbJ_PwPy8MINLO_ade.csv', nrows=10)

# Load the Data

def StopDataLoader(path, features, test="550_520", selection="", treename="bdttree", suffix="", signal="DM30", fraction=1.0, useSF=False):
  if signal not in signalMap:
    raise KeyError("Unknown training signal requested ("+signal+")")
  if test not in signalMap:
    raise KeyError("Unknown test signal requested ("+test+")")
  if fraction >= 1.0:
    fraction = 1.0
  if fraction < 0.0:
    raise ValueError("An invalid fraction was chosen")
  if "XS" not in features:
    features.append("XS")
  if "Nevt" not in features:
    features.append("Nevt")
  if "Event" not in features:
    features.append("Event")
  if "weight" not in features:
    features.append("weight")

  # Train and Test Data split for Signal

  sigDev = None
  sigVal = None


  testPath = "nTuples16_v2017-10-19_test"+suffix+"/"
  trainPath = "nTuples16_v2017-10-19_train"+suffix+"/"

  #testPath = "test/"
  #trainPath = "train/"

  for sigName_test in signalMap[test]:
    tmp = root_numpy.root2array(
                                path + testPath + sigName_test + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if sigVal is None:
      sigVal = pandas.DataFrame(tmp)
    else:
      sigVal = sigVal.append(pandas.DataFrame(tmp), ignore_index=True)


  for sigName in signalMap[signal]:
    tmp = root_numpy.root2array(
                                path + trainPath + sigName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if sigDev is None:
      sigDev = pandas.DataFrame(tmp)
    else:
      sigDev = sigDev.append(pandas.DataFrame(tmp), ignore_index=True)

  # Train and Test Data split for Background

  bkgDev = None
  bkgVal = None
  for bkgName in bkgDatasets:
    tmp = root_numpy.root2array(
                                path + trainPath + bkgName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if bkgDev is None:
      bkgDev = pandas.DataFrame(tmp)
    else:
      bkgDev = bkgDev.append(pandas.DataFrame(tmp), ignore_index=True)

    tmp = root_numpy.root2array(
                                path + testPath + bkgName + suffix + ".root",
                                treename=treename,
                                selection=selection,
                                branches=features
                                )
    if fraction < 1.0:
      tmp = tmp[:int(len(tmp)*fraction)]
    if bkgVal is None:
      bkgVal = pandas.DataFrame(tmp)
    else:
      bkgVal = bkgVal.append(pandas.DataFrame(tmp), ignore_index=True)

  # Data Labelling

  sigDev["category"] = 1
  sigVal["category"] = 1
  bkgDev["category"] = 0
  bkgVal["category"] = 0
  sigDev["sampleWeight"] = 1
  sigVal["sampleWeight"] = 1
  bkgDev["sampleWeight"] = 1
  bkgVal["sampleWeight"] = 1

  if fraction < 1.0:
    sigDev.weight = sigDev.weight/fraction
    sigVal.weight = sigVal.weight/fraction
    bkgDev.weight = bkgDev.weight/fraction
    bkgVal.weight = bkgVal.weight/fraction

  if not useSF:
    sigDev.sampleWeight = sigDev.weight
    sigVal.sampleWeight = sigVal.weight
    bkgDev.sampleWeight = bkgDev.weight
    bkgVal.sampleWeight = bkgVal.weight
  else:
    scale = fraction if fraction < 1.0 else 1.0
    sigDev.sampleWeight = 1/(sigDev.Nevt*scale)
    sigVal.sampleWeight = 1/(sigVal.Nevt*scale)
    bkgDev.sampleWeight = bkgDev.XS/(bkgDev.Nevt*scale)
    bkgVal.sampleWeight = bkgVal.XS/(bkgVal.Nevt*scale)

  sigDev.sampleWeight = sigDev.sampleWeight/sigDev.sampleWeight.sum()
  sigVal.sampleWeight = sigVal.sampleWeight/sigVal.sampleWeight.sum()
  bkgDev.sampleWeight = bkgDev.sampleWeight/bkgDev.sampleWeight.sum()
  bkgVal.sampleWeight = bkgVal.sampleWeight/bkgVal.sampleWeight.sum()

  dev = sigDev.copy()
  dev = dev.append(bkgDev.copy(), ignore_index=True)
  val = sigVal.copy()
  val = val.append(bkgVal.copy(), ignore_index=True)

  return dev, val
