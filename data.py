import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

nrows_sinal = 991141
nrows_stopWt = 277816
nrows_ttbar = 4168037
nrows_Wjets = 16650877
nrows_WlvZqq = 188395
nrows_WqqWlv = 334495

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

print "Reading time: ", (time.time() - start), "s  of", i, "% of data"

del df_sinal, df_stopWt, df_ttbar, df_WlvZqq, df_WqqWlv, df_WJets
