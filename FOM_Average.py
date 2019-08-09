import os
import localConfig as cfg
import numpy as np
import matplotlib.pyplot as plt
from commonFunctions import assure_path_exists
from shutil import copyfile

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chose models to plot fom")
    parser.add_argument('-i', '--version', type=str, required=True, help="Select the model to average and plot")
    parser.add_argument('-env', '--env', action='store_true',  help="Used to draw an envelope in the average plot")

    args = parser.parse_args()
    version = args.version
    list = version.split()

    # Creating a list with filepaths:
    # lgbk = "/home/t3atlas/ev19u056/projetoWH/"
    testpath = cfg.lgbk + "test/"
    i = 0
    path = []
    while (i < len(list)) :
        path.append(testpath+"Model_Ver_"+str(list[i])+"/plots_Model_Ver_"+str(list[i])+"/")
        i += 1

    #Importing data in lists used to compute average:
    i = 0
    list_Evo = []
    while (i < len(path)):
        list_Evo.append(np.loadtxt(path[i]+"FOM_evo_data.txt",delimiter="\n"))
        i += 1

    #Initializing
    i = 0
    list_ave_Evo = []
    while (i < len(list_Evo[1])):
        list_ave_Evo.append(0.0)
        i += 1

    #Computing average
    i = 0
    while (i < len(list_Evo[1])):
        a = 0
        ave_Evo=0
        while (a < len(list)):
            ave_Evo = ave_Evo + list_Evo[a][i]
            a += 1
        list_ave_Evo[i] = ave_Evo/len(list)
        i += 1

    #Envelope:
    if args.env:
        #Initializing envelope
        list_max_env = []
        list_min_env = []
        i = 0
        while i < len((list_Evo[1])):
            list_max_env.append(0.0)
            list_min_env.append(0.0)
            i = i + 1
        #taking envelope's values
        i = 0
        while (i < len(list_Evo[1])):
            a = 0
            b = 0
            c = 100000000
            while (a < len(list)):
                if list_Evo[a][i] > b :
                    b = list_Evo[a][i]
                if list_Evo[a][i] < c :
                    c = list_Evo[a][i]
                a = a + 1
            list_max_env[i] = b
            list_min_env[i] = c
            i = i + 1


    #Creating a new file where average will be stored:
    averagepath = testpath + "Model_Ver_" + version[:2] + "_average/plots_Model_Ver_" + version[:2] + "_average/"

    assure_path_exists(testpath + "Model_Ver_" + version[:2] + "_average/")
    assure_path_exists(testpath + "Model_Ver_" + version[:2] + "_average/plots_Model_Ver_" + version[:2] + "_average/")

    f = open(averagepath + "FOM_evo_data.txt","w+")
    f.write("\n".join(map(str,list_ave_Evo)))
    f.close()

    copyfile(testpath + "Model_Ver_10/plots_Model_Ver_10/FOM_cut_data.txt", averagepath + "FOM_cut_data.txt")

    print("Model_Ver_" + version[:2] + "-average saved! :-)")

    if args.env:
        f = open(averagepath + "FOM_max_data.txt","w+")
        f.write("\n".join(map(str,list_max_env)))
        f.close()

        f = open(averagepath + "FOM_min_data.txt","w+")
        f.write("\n".join(map(str,list_min_env)))
        f.close()

        print("Envelope saved in Model_Ver_" + version[:2] + "_average/" +":D")
