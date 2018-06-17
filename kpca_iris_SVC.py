import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
#df.to_csv('iris.csv')
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC         # 'rbf'
import time as t
from sklearn.model_selection import train_test_split

# load dataset into Pandas DataFrame
def SVC_With_Different_Kernels(data_before_KPCA,data_after_KPCA,ker):
    print('* NOTE:  Dimensions are reduced using KPCA with kernel:'+str(ker))
    df = pd.read_csv(data_before_KPCA)
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # Separating out the features
    X = df.loc[:, features].values
    # Separating out the target
    Y = df.loc[:,['target']].values
    X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size = 0.3, random_state = 100)
    y_train=y_train.ravel()
    y_test=y_test.ravel()
    #classifier.fit(X_train, y_train.squeeze())
    s=t.time()
    svm_model_linear = SVC(kernel ='linear', C = 1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)
    e=t.time()
    # model accuracy for X_test  
    accuracy = svm_model_linear.score(X_test, y_test)
    print('*'*11,' SVC Kernel : LINEAR ','*'*11)
    print('Time taken for classification is :',e-s )
    print('Accuracy of IRIS Dataset Before KPCA using SVC is :',accuracy*100)

    #============================================================================

    df = pd.read_csv(data_after_KPCA)
    #df.to_csv('iris.csv')
    from sklearn.preprocessing import StandardScaler

    features = ['PC-1', 'PC-2']
    # Separating out the features
    X = df.loc[:, features].values
    # Separating out the target
    Y = df.loc[:,['target']].values
    X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size = 0.3, random_state = 100)
    y_train=y_train.ravel()
    y_test=y_test.ravel()
    s=t.time()
    svm_model_linear = SVC(kernel ='linear', C = 1).fit(X_train, y_train)

    svm_predictions = svm_model_linear.predict(X_test)
    e=t.time()
    # model accuracy for X_test  
    accuracy = svm_model_linear.score(X_test, y_test)
    print('Time taken for classification is :',e-s )
    print('Accuracy of IRIS Dataset After KPCA using SVC is :',accuracy*100,'\n')




data_before_KPCA="D:\Python_programs\ML\Iris Data\KPCA\iris.csv"
files=['D:\Python_programs\ML\Iris Data\KPCA\iris_after_KPCA_using_linear.csv',
       'D:\Python_programs\ML\Iris Data\KPCA\iris_after_KPCA_using_rbf.csv',
       'D:\Python_programs\ML\Iris Data\KPCA\iris_after_KPCA_using_poly.csv']
ker=['linear','rbf','poly']
for i in range(3):
    SVC_With_Different_Kernels(data_before_KPCA,files[i],ker[i])
