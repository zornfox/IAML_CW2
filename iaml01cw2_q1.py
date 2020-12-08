


#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.
import os
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from helpers.iaml01cw2_helpers import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#<----

# Q1.1
def iaml01cw2_q1_1():
    # normalisation start
    global Xtrn, Ytrn, Xtst, Ytst, Xtrn_orig, Xtst_orig, Xtrn, Xtst, Xmean, Xtrn_nm, Xtst_nm
    DataPath = os.path.join(os.getcwd(), 'data')
    Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(DataPath)
    # make a back up (nomalisation step1)
    Xtrn_orig = np.copy(Xtrn)
    Xtst_orig = np.copy(Xtst)
    # over-wirtten Xtrn and Xtst (nomalisation step2)
    Xtrn = Xtrn/255.0
    Xtst = Xtst/255.0
    # Calculate mean Value (nomalisation step3)
    Xmean = np.mean(Xtrn, axis=0)
    # Subtract Xmean from each row of Xtrn and Xtst (nomalisation step4)
    Xtrn_nm = Xtrn - Xmean
    Xtst_nm = Xtst - Xmean
    return Xtrn_nm[0, :4], Xtrn_nm[-1, :4]
iaml01cw2_q1_1()   

# Q1.2
def iaml01cw2_q1_2():
    fig, axes = plt.subplots(10,5)
    fig.set_size_inches(15,25)
    sample_list=[0,1,-2,-1]
    image_position=[1,2,3,4]
    dic = dict(zip(image_position,sample_list))
    #print(dic)
    for i in range(10):
        class_index = np.where(Ytrn == i)
        #print(class_index[0])
        sample = Xtrn[class_index]
        sample_mean = np.mean(sample, axis=0)
        image=sample_mean.reshape(28,28)
        Edis = [np.sum((p-sample_mean)**2) for p in sample]
        # return the index of sorted Edistance
        index = [x for x,y in sorted(enumerate(Edis), key = lambda x: x[1])]
        axes[i,0].imshow(image, cmap="gray_r")
        # plot the image of mean 
        axes[i,0].set_title("[class={},mean]".format(i))
        axes[i,0].axis('off')
        # plot 2 closest sample and 2 further samples
        for key, value in dic.items():
            samp = sample[index[value]]
            imag = samp.reshape(28,28)
            axes[i, key].imshow(imag, cmap='gray_r')
            axes[i, key].set_title("[class={},sample={}]".format(i, class_index[0][index[value]]))   
            axes[i, key].axis('off')
        plt.savefig('q1_2.png')
iaml01cw2_q1_2()   

# Q1.3
def iaml01cw2_q1_3():
    X_pca = PCA(n_components=784)
    X_pca.fit(Xtrn_nm)
    F5= X_pca.explained_variance_[0:5].tolist()
    Round_F5=[round(i,3) for i in F5]
    return Round_F5
iaml01cw2_q1_3()   


# Q1.4
def iaml01cw2_q1_4():
    X_pca = PCA(n_components=784)
    X_pca.fit(Xtrn_nm)
    plt.figure(figsize=(12,8))
    plt.plot(np.cumsum(X_pca.explained_variance_ratio_))
    my_x_ticks = np.arange(1, 785,87)
    plt.xticks(my_x_ticks)
    plt.title('Cumulative Explained Variance Ratio VS Number of componets')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance ratio')
    plt.savefig('q1_4.png')
    return
iaml01cw2_q1_4()   


# Q1.5
def iaml01cw2_q1_5():
    X_pca = PCA(n_components=784)
    X_pca.fit(Xtrn_nm)
    fig, axes = plt.subplots(2,5, figsize=(10,5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_pca.components_[i].reshape(28,28), cmap='gray_r')
        ax.set_title("[PC={}]".format(i+1))
        ax.axis('off')
    plt.savefig('q1_5.png')
iaml01cw2_q1_5()   

# Q1.6
def iaml01cw2_q1_6():
    K = [5,20,50,200]
    for k in K:
        pca_test = PCA(n_components=k)
        pca_test.fit(Xtrn_nm)
        for i in range(10):
            class_index=np.where(Ytrn == i)
            origin = Xtrn_nm[class_index][0,:].reshape(1,-1)
            recstr = pca_test.inverse_transform(pca_test.transform(Xtrn_nm[class_index][0,:].reshape(1,-1)))
            rmse = np.sqrt(mean_squared_error(origin,recstr))
            print(float("%0.3f" % (rmse)))
        print("\n")
    return
iaml01cw2_q1_6()   


# Q1.7
def iaml01cw2_q1_7():
    K=[5,20,50,200]
    Pic_List=[]
    for k in K:
        pca_newcomp = PCA(n_components=k)
        pca_newcomp.fit(Xtrn_nm)
        for i in range(10):
            class_index=np.where(Ytrn == i)
            origin = Xtrn_nm[class_index][0,:].reshape(1,-1)
            recstr = pca_newcomp.inverse_transform(pca_newcomp.transform(Xtrn_nm[class_index][0,:].reshape(1,-1)))
            pic = recstr+Xmean
            Pic_List.append(pic)
    Pic_List=np.array(Pic_List)
    fig, axes = plt.subplots(10,4)
    fig.set_size_inches(15,25)
    index=0
    for i in range(4):
        for j in range(10):
            axes[j,i].imshow(Pic_List[index].reshape(28,28), cmap='gray_r')
            axes[j,i].set_title("[class={}, K={}]".format(j,K[i]))
            axes[j,i].axis('off')
            index+=1
    plt.savefig('q1_7.png')
iaml01cw2_q1_7()  


# Q1.8
def iaml01cw2_q1_8():
    pca=PCA(n_components=2)
    PCA_2D= pca.fit_transform(Xtrn_nm)
    Color_sequence= Ytrn
    plt.figure(figsize=(10,8))
    plt.scatter(PCA_2D[:,0], PCA_2D[:,1],c= Color_sequence, lw=2,cmap='coolwarm', alpha=1, s=1)
    plt.title("2D PCA plane")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()
    plt.savefig('q1_8.png')
iaml01cw2_q1_8()   
