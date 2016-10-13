# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


def lissage1(df_in, size, variable):
    """
    PARAMETRES:

    **df_in** (*pandas dataframe*): dataframe \n
    **size** (*int impair*): dimension de la fenetre \n
    **variableslist** (*list*): liste des variables a traiter \n
    **variable** (*string*): variable de reference pour modifier les variables suivantes sur les memes indices

    Renvoie une dataframe avec les memes dimensions

    """
    assert (size % 2 == 1), "La taille de la fenetre doit etre impaire"

    dataframe = df_in.copy()
    dataframe[variable] = median_filter(dataframe[variable].values, size=size)
    nonmodif_idx = np.where(df_in[variable].values == dataframe[variable])[0]  # recherche des vaindices de valeurs non modifiees
    modif_idx = np.where(df_in[variable].values != dataframe[variable])[0]  # recherche des indices de valeurs modifiees
    for v in list(set(df_in.columns) - set([variable])):
        mat = dataframe[v].values[:]
        mat_out = np.zeros(mat.shape[0])
        mat_out[:] = np.nan
        for idx in modif_idx:  #median RuntimeWarning
            if idx-((size-1)/2) < 0:
                diff_valeurs = np.abs(idx - (size / 2) )
                mat_tmp = np.append(mat[:idx+(size / 2)+1], mat[-diff_valeurs:])  # rajout de valeurs de fin de matrice en debut pour eviter valeurs nulles
                mat_out[idx] = np.median(mat_tmp)
            elif idx + ((size - 1) / 2) + 1 > mat.shape[0]:
                diff_valeurs = idx+(size/2)+1 - mat.shape[0]
                mat_tmp = np.append(mat[idx - (size / 2):], mat[:diff_valeurs])  # rajout de valeurs de debut de matrice en fin pour eviter valeurs nulles
                mat_out[idx] = np.median(mat_tmp)
            else:
                mat_out[idx] = np.median(mat[idx-(size/2):idx+(size/2)+1])
            mat_out[nonmodif_idx] = mat[nonmodif_idx]
        dataframe[v] = mat_out[:]
    return dataframe


def argMedian(a):
    m = np.median(a)
    ind = np.argsort(a)[(len(a)/2)+1]
    return m, ind


def lissage(matrice, ww, lref):
    arr = matrice.copy()
    lo = range(arr.shape[1])
    lo.remove(lref)
    for i in range(ww/2,arr.shape[0]-(ww/2)):
        med, ix = argMedian(arr[i-((ww-1)/2):i+((ww-1)/2)+1, lref]) # mediane et indice(dans la fenetre de 9 valeurs) de valeur utilisée pour remplacer la valeur cible [i] de la couche de ref
        if arr[i, lref] != med:
            print 'changes'
            arr[i, lref] = med
            # modif des valeurs des couches suivantes en utilisant les mêmes indices
            arr[i,lo[0]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[0]][ix] 
            arr[i,lo[1]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[1]][ix]
            arr[i,lo[2]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[2]][ix]
    return arr


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################


if __name__ == '__main__':

    w_size = 15
    df = pd.DataFrame(np.random.rand(100),columns=['valeurs1'])
    df['valeurs2'] = np.random.rand(100)
    dfo = lissage(df,w_size,['valeurs1','valeurs2'],'valeurs1')
    
    plt.plot(df.valeurs1,'k-',label='valeurs1')
    plt.plot(df.valeurs2,'r-', label='valeurs2')
    plt.plot(dfo.valeurs1,'b--*', label='valeurs1 lissees')
    plt.plot(dfo.valeurs2, 'g--*', label='valeurs2 lissees')
    plt.legend()
    