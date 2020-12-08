

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.
import os
import numpy as np
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from helpers.iaml01cw2_helpers import *

#<----

# Q2.1
def iaml01cw2_q2_1():
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
    # normalisation end
    
    MLR=LogisticRegression()
    MLR.fit(Xtrn_nm, Ytrn)
    Pred= MLR.predict(Xtst_nm)
    Cm = confusion_matrix(Ytst, Pred)
    score = MLR.score(Xtst_nm, Ytst)
    return Cm, score
iaml01cw2_q2_1()  

# Q2.2
def iaml01cw2_q2_2():
    SVM = SVC(kernel='rbf', C=1.0, gamma='auto')
    SVM.fit(Xtrn_nm, Ytrn)
    Pred=SVM.predict(Xtst_nm)
    Cm=confusion_matrix(Ytst, Pred)
    score=SVM.score(Xtst_nm, Ytst)
    return Cm, score
iaml01cw2_q2_2()   

# Q2.3
def iaml01cw2_q2_3():
    X_pca = PCA(n_components=784)
    X_pca.fit(Xtrn_nm)
    P= X_pca.transform(Xtrn_nm)
    #get the standarded diviations of 1st and 2nd components   
    q1=np.sqrt(X_pca.explained_variance_[0])
    print(q1)
    q2=np.sqrt(X_pca.explained_variance_[1])
    print(q2)
    pp=np.mgrid[-5*q1:5*q1:100j,-5*q2:5*q2:100j]
    x,y=pp
    #become flat
    flat_x=x.ravel()
    flat_y=y.ravel()
    #projected point z without inverse transform    
    z=(np.c_[flat_x,flat_y]).tolist()
    #points on the 2D-plane spanned with the first two PC without transform
    z_2D=[]
    zeros=(np.zeros(784-2, dtype=int)).tolist()
    for i in z:
        concatenate= i + zeros
        z_2D.append(concatenate)

    MLR=LogisticRegression()
    MLR.fit(Xtrn_nm, Ytrn)
    inverseT_z=(MLR.predict(X_pca.inverse_transform(np.array(z_2D)))).reshape(100,100)
    plt.figure(figsize=(10,8))
    plt.contourf(x,y,inverseT_z, cmap='coolwarm')
    plt.colorbar()
    plt.title("Decision regions for the logistic regression classfier")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig('q2_3.png')
    
iaml01cw2_q2_3()   

# Q2.4
def iaml01cw2_q2_4():
    X_pca = PCA(n_components=784)
    X_pca.fit(Xtrn_nm)
    P= X_pca.transform(Xtrn_nm)
    #get the standarded diviations of 1st and 2nd components   
    q1=np.sqrt(X_pca.explained_variance_[0])
    print(q1)
    q2=np.sqrt(X_pca.explained_variance_[1])
    print(q2)
    pp=np.mgrid[-5*q1:5*q1:100j,-5*q2:5*q2:100j]
    x,y=pp
    #become flat
    flat_x=x.ravel()
    flat_y=y.ravel()
    #projected point z without inverse transform    
    z=(np.c_[flat_x,flat_y]).tolist()
    #points on the 2D-plane spanned with the first two PC without transform
    z_2D=[]
    zeros=(np.zeros(784-2, dtype=int)).tolist()
    for i in z:
        concatenate= i + zeros
        z_2D.append(concatenate)

    SVM = SVC(kernel='rbf', C=1.0, gamma='auto')
    SVM.fit(Xtrn_nm, Ytrn)
    inverseT_z=(SVM.predict(X_pca.inverse_transform(np.array(z_2D)))).reshape(100,100)
    plt.figure(figsize=(10,8))
    plt.contourf(x,y,inverseT_z, 10, cmap='coolwarm')
    plt.colorbar()
    plt.title("Decision regions for the SVM classfier")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
iaml01cw2_q2_4()   

# Q2.5
def iaml01cw2_q2_5():
    global Accuracy_list,C
    Class_index=[]
    Accuracy_list=[]
    for i in range(10):
        index= np.where(Ytrn==i)
        Class_index.append(index[0].tolist())
    Class_index_reduce=[]
    for item in Class_index:
        Class_index_reduce.append(item[0:1000])
    Merge_class_index=[j for i in Class_index_reduce for j in i]
    Ysamll=Ytrn[Merge_class_index]
    Xsamll=Xtrn[Merge_class_index,:]
#     C=[]
#     step=(10**3-10**(-2))/9
#     for i in range(10):
#         C.append(10**(-2)+step*i)
    C=np.logspace(-2.0,3.0, num=10)
    for each in C:
        model=SVC(kernel='rbf', C=each, gamma='auto')
        accuracy = np.mean(cross_val_score(model, Xsamll, Ysamll, scoring='accuracy', cv = 3)) 
        Accuracy_list.append(accuracy)
#     model=SVC(kernel='rbf', C=0.01, gamma='auto')
#     accuracy = np.mean(cross_val_score(model, Xsamll, Ysamll, cv = 3)) 
#     return accuracy
    C_max = C[Accuracy_list.index(max(Accuracy_list))]
    fig=plt.figure(figsize=(10,8))
    cross_val_score
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.scatter(C, Accuracy_list)
    plt.xlabel('the regularisation parameter C')
    plt.ylabel('mean cross-validation classfication accuracy')
    plt.savefig('q2_5.png')
    return max(Accuracy_list), C_max
iaml01cw2_q2_5()   

# Q2.6 
def iaml01cw2_q2_6():
    svmOp=SVC(C=C[Accuracy_list.index(max(Accuracy_list))]).fit(Xtrn_nm, Ytrn)
    Training_acc=svmOp.score(Xtrn_nm, Ytrn)
    Testing_acc=svmOp.score(Xtst_nm, Ytst)
    return Training_acc, Testing_acc
# iaml01cw2_q2_6()   # comment this out when you run the function

