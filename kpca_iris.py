import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load dataset into Pandas DataFrame
df = pd.read_csv("D:\Python_programs\ML\Iris Data\KPCA\iris.csv")
#df.to_csv('iris.csv')


from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import KernelPCA


## Finding the principle components
#   KERNELS : linear,rbf,poly
#

def Kernel_Pca(ker):
    kpca = KernelPCA(n_components=4, kernel=ker, gamma=15)
    x_kpca = kpca.fit_transform(x)
    kpca_transform = kpca.fit_transform(x)
    explained_variance = np.var(kpca_transform, axis=0)
    ev = explained_variance / np.sum(explained_variance)

    #--------- Bar Graph for Explained Variance Ratio ------------
    plt.bar([1,2,3,4],list(ev*100),label='Principal Components',color='b')
    plt.legend()
    plt.xlabel('Principal Components ')
    #----------------------
    n=list(ev*100)
    pc=[]
    for i in range(len(n)):
            n[i]=round(n[i],4)
            pc.append('PC-'+str(i+1)+'('+str(n[i])+')')

    #----------------------
    plt.xticks([1,2,3,4],pc, fontsize=7, rotation=30)
    plt.ylabel('Variance Ratio')
    plt.title('Variance Ratio of IRIS Dataset using kernel:'+str(ker))
    plt.show()
    #---------------------------------------------------
    # *Since the initial 2 principal components have high variance.
    #   so, we select pc-1 and pc-2.
    #---------------------------------------------------
    kpca = KernelPCA(n_components=2, kernel=ker, gamma=15)
    x_kpca = kpca.fit_transform(x)
    principalComponents = kpca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['PC-1', 'PC-2'])
    # Adding lables
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
    # Plotting pc1 & pc2
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC-1', fontsize = 15)
    ax.set_ylabel('PC-2', fontsize = 15)
    ax.set_title('KPCA on IRIS Dataset using kernel:'+str(ker), fontsize = 20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC-1']
                   , finalDf.loc[indicesToKeep, 'PC-2']
                   , c = color
                   , s = 30)
    ax.legend(targets)
    ax.grid()
    plt.show() # FOR SHOWING THE PLOT
    #------------------- SAVING DATA INTO CSV FILE ------------
    finalDf.to_csv('iris_after_KPCA_using_'+str(ker)+'.csv')


#------------------------------------------------------
k=['linear','rbf','poly']
for i in k:
    Kernel_Pca(i)
