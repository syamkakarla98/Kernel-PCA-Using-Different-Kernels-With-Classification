
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


df = pd.read_csv('D:\Python_programs\ML\Iris Data\KPCA\iris_after_KPCA_using_poly.csv')
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
svm_model = SVC(kernel ='rbf', C = 1).fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)
e=t.time()
 # model accuracy for X_test  
accuracy = svm_model.score(X_test, y_test)
print('Time taken for classification is :',e-s )
print('Accuracy of IRIS Dataset After KPCA using SVC is :',accuracy*100)
