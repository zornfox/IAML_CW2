

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.cluster import hierarchy 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn.decomposition import PCA
from helpers.iaml01cw2_helpers import *

#<----

# Q3.1
def iaml01cw2_q3_1():
    global Xtrn, Ytrn, Xtst, Ytst, Km
    DataPath = os.path.join(os.getcwd(), 'data')
    Xtrn, Ytrn, Xtst, Ytst = load_CoVoST2(DataPath)
    Km = KMeans(n_clusters=22, random_state=1).fit(Xtrn)
    alldistance = Km.transform(Xtrn)
    totalDis = np.min(alldistance**2, axis=1).sum()
    mydict = {i: np.where(Km.labels_ == i)[0] for i in range(Km.n_clusters)}
    dictlist = []
    for key, value in mydict.items():
        temp = [key,value.shape]
        dictlist.append(temp)
    return totalDis,dictlist
iaml01cw2_q3_1()   

# Q3.2
def iaml01cw2_q3_2():
    global mean_vector_list,MeanV_2D,Cls
    mean_vector_list=[]
    for i in range(22):
        mean_vector=(np.mean(Xtrn[Ytrn==i], axis=0))
        mean_vector_list.append(mean_vector)
#     print(np.array(mean_vector_list))
    PCA_2D = PCA(n_components=2)
    MeanV_2D=PCA_2D.fit_transform(mean_vec)
    Cls=PCA_2D.fit_transform(Km.cluster_centers_)
#     print(Cls)
#     print(MeanV_2D)
#     print(Cls)
#     fig, axes = plt.subplots()
#     colors=matplotlib.cm.rainbow(np.linspace(0,1,len(R)))
#     cs = [colors[i] for i in range(22)]
#     axes.scatter(R[:,0], R[:,1], color=cs)
#     axes.legend()
    colors = ['black','slategray','silver','rosybrown', 'peru','firebrick','red','sienna','peachpuff',
              'goldenrod','orange','darkkhaki','yellow', 'lime', 'mediumseagreen','cyan','deepskyblue', 
              'blue', 'blueviolet', 'plum', 'purple','orchid']
    Mean_V_labels=['Class0: Arabic(Ar)-Mean Vector','Class1: Catalan(Ca)- Mean Vector','Class2: Welsh(Cy)- Mean Vector','Class3: German(De)- Mean Vector',
                    'Class4: English(En)- Mean Vector','Class5: Spanish(Es)- Mean Vector','Class6: Estonian(Et)- Mean Vector','Class7: Persian(Fa)- Mean Vector',
                    'Class8: French(Fr)- Mean Vector','Class9: Indonesian(Id)- Mean Vector','Class10: Italian(It)- Mean Vector','Class11: Japanese(Ja)- Mean Vector',
                    'Class12: Latvian(Lv)- Mean Vector','Class13: Mongolian(Mn)- Mean Vector','Class14: Dutch(Nl)- Mean Vector','Class15: Russian(Ru)- Mean Vector',
                    'Class16: Slovenian(Sl)- Mean Vector','Class17: Swedish(Sv)- Mean Vector','Class18: Portuguese(Pt)- Mean Vector','Class19: Tamil(Ta)- Mean Vector',
                    'Class20: Turkish(Tr)- Mean Vector','Class21: Chinese(Zh)- Mean Vector']
    Cluster_C_labels=['Class0: Arabic(Ar)- Cluster Centre','Class1: Catalan(Ca)- Cluster Centre','Class2: Welsh(Cy)- Cluster Centre','Class3: German(De)- Cluster Centre',
                    'Class4: English(En)- Cluster Centre','Class5: Spanish(Es)- Cluster Centre','Class6: Estonian(Et)- Cluster Centre','Class7: Persian(Fa)- Cluster Centre',
                    'Class8: French(Fr)- Cluster Centre','Class9: Indonesian(Id)- Cluster Centre','Class10: Italian(It)- Cluster Centre','Class11: Japanese(Ja)- Cluster Centre',
                    'Class12: Latvian(Lv)- Cluster Centre','Class13: Mongolian(Mn)- Cluster Centre','Class14: Dutch(Nl)- Cluster Centre','Class15: Russian(Ru)- Cluster Centre',
                    'Class16: Slovenian(Sl)- Cluster Centre','Class17: Swedish(Sv)- Cluster Centre','Class18: Portuguese(Pt)- Cluster Centre','Class19: Tamil(Ta)- Cluster Centre',
                    'Class20: Turkish(Tr)- Cluster Centre','Class21: Chinese(Zh)- Cluster Centre']
    plt.figure(figsize=(12,10)) 
    for i in range(22):
        plt.scatter(MeanV_2D[i,0], MeanV_2D[i,1], c=colors[i], s=60,  label=Mean_V_labels[i])
        plt.scatter(Cls[i,0], Cls[i,1], c=colors[i], marker='x',linewidths=3, s=60,label=Cluster_C_labels[i])
    plt.legend(loc=[1, 0])
    plt.title("Mean vectors and cluster centres for all the 22 languages on a 2D-PCA plane")
    plt.xlabel('PC1')
    plt.ylabel('PC1')
    plt.savefig('q3_2.png')

iaml01cw2_q3_2()   # comment this out when you run the function

# Q3.3
def iaml01cw2_q3_3():
    Z=hierarchy.linkage(mean_vector_list, 'ward')
    fig = plt.figure(figsize=(10, 6))
    dn = hierarchy.dendrogram(Z, orientation='right')
    plt.title("Hierarchical Clustering")
    plt.xlabel("Distance")
    plt.ylabel("language class")
iaml01cw2_q3_3()

# Q3.4
def iaml01cw2_q3_4():
    vec66=[]
    for i in range(22):
        Km_new = KMeans(n_clusters=3, random_state=1).fit(Xtrn[Ytrn==i])
        Km_new_cluster_centre=Km_new.cluster_centers_
        vec66.append(Km_new_cluster_centre)
    reform_vec66=[item for j in vec66 for item in j]
    Z1=hierarchy.linkage(reform_vec66, 'ward')
    fig1 = plt.figure(figsize=(10, 6))
    dn1 = hierarchy.dendrogram(Z1, orientation='right')
    plt.title("Hierarchical Clustering with linkage method of ward")
    plt.xlabel("Distance")
    plt.ylabel("language class")
    Z2=hierarchy.linkage(reform_vec66, 'single')
    fig2 = plt.figure(figsize=(10, 6))
    dn2 = hierarchy.dendrogram(Z2, orientation='right')
    plt.title("Hierarchical Clustering with linkage method of single")
    plt.xlabel("Distance")
    plt.ylabel("language class")
    Z3=hierarchy.linkage(reform_vec66, 'complete')
    fig3 = plt.figure(figsize=(10, 6))
    dn3 = hierarchy.dendrogram(Z3, orientation='right')
    plt.title("Hierarchical Clustering with linkage method of complete")
    plt.xlabel("Distance")
    plt.ylabel("language class")
    
iaml01cw2_q3_4()

# Q3.5
def iaml01cw2_q3_5():
    L0_trn = Xtrn[Ytrn==0]
    L0_tst = Xtst[Ytst==0]
    K = [1,3,5,10,15]
    Diag_Trn=[]
    Diag_Tst=[]
    Full_Trn=[]
    Full_Tst=[]
    for k in K:
        gm1=GaussianMixture(n_components=k, covariance_type='diag').fit(L0_trn)
        Diag_Trn.append(round(gm1.score(L0_trn),3))
        Diag_Tst.append(round(gm1.score(L0_tst),3))
        gm2=GaussianMixture(n_components=k, covariance_type='full').fit(L0_trn)
        Full_Trn.append(round(gm2.score(L0_trn),3))
        Full_Tst.append(round(gm2.score(L0_tst),3))
    plt.figure(figsize=(10,8))
    plt.plot(K, Diag_Trn, '-o', color='red', label='Diag_Trn')
    plt.plot(K, Diag_Tst, '-o', color='blue', label='Diag_Tst')
    plt.plot(K, Full_Trn, '-o', color='green', label='Full_Trn')
    plt.plot(K, Full_Tst, '-o', color='yellow', label='Full_Tst')
    my_x_ticks = np.arange(1, 16,2)
    plt.xticks(my_x_ticks)
    plt.xlabel("Number of mixture components: K")
    plt.ylabel("Per-sample average log-likelihood")
    plt.legend()
    plt.savefig('q3_5.png')
    return Diag_Trn, Diag_Tst, Full_Trn, Full_Tst
    
iaml01cw2_q3_5()

