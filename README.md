# [Kernel PCA ](https://scholar.google.co.in/scholar?q=kernel+pca+dimensionality+reduction&hl=en&as_sdt=0&as_vis=1&oi=scholart) using Different Kernels With Classification using python 

   
### Prerequisites

The things that you must have a decent knowledge on: 
```
    * Python
    * Linear Algebra
```

### Installation

* This project is fully based on python. So, the necessary modules needed for computaion are:
```
    * Numpy
    * Sklearm
    * Matplotlib
    * Pandas
```
* The commands needed for installing the above modules on windows platfom are:
```python

    pip install numpy
    pip install sklearn
    pip install matplotlib
    pip install pandas
```
* we can verify the installation of modules by  importing the modules. For example:
```python

    import numpy
    from sklearn.decomposition import kernelPCA 
    import matplotlib.pyplot as plt
    import pandas as pd
```
### Explanation 

* We are performing the the **dimensionality reduction**  using [**Kernel PCA**](http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html)  with three different Kernels: 
     * [Linear](https://stats.stackexchange.com/questions/101344/is-kernel-pca-with-linear-kernel-equivalent-to-standard-pca)
     * [Radial Basis Function(RBF)](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
     * [Polynomial](https://en.wikipedia.org/wiki/Polynomial_kernel)
 * Here we are performing the operations on the [IRIS Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
     

1. The output of kernel PCA with **_Linear_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Linear_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      
      ![vr_using_linear_kernel](https://user-images.githubusercontent.com/36328597/41507798-9d33a58a-7257-11e8-95c1-81b31b0659f8.png)

      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * The scatter plot for the **2 Pricipal Components** is :

      
      ![kpca_using_linear_kernel](https://user-images.githubusercontent.com/36328597/41507794-9c0983e6-7257-11e8-8b2d-1137c3e272b8.png)



   * The dimensionally reduced file is saved to [iris_after_KPCA_using_linear.csv](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/iris_after_KPCA_using_linear.csv).
 
2. The output of kernel PCA with **_Radial Basis Function(RBF)_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Radial Basis Function(RBF)_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      ![vr_using_rbf_kernel](https://user-images.githubusercontent.com/36328597/41507821-f39ee5c4-7257-11e8-99fc-b90548ff26c3.png)

      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * THe scatter plot for the **2 Pricipal Components** is :

      
      ![kpca_using_rbf_kernel](https://user-images.githubusercontent.com/36328597/41507820-f36374ee-7257-11e8-8459-09ed3b43d1ae.JPG)


      * The dimensionally reduced file is saved to [iris_after_KPCA_using_rbf.csv](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/iris_after_KPCA_using_poly.csv).    
   
3. The output of kernel PCA with **_Polynomial_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Polynomial_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      ![vr_using_poly_kernel](https://user-images.githubusercontent.com/36328597/41507799-9d72765c-7257-11e8-8564-20e07c526b3f.png)

      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * The scatter plot for the **2 Pricipal Components** is :

      ![kpca_using_poly_kernel](https://user-images.githubusercontent.com/36328597/41507795-9c73fdac-7257-11e8-9e61-8deddc25203d.JPG)



   * The dimensionally reduced file is saved to [iris_after_KPCA_using_poly.csv](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/iris_after_KPCA_using_rbf.csv).
   
   
### Classification

   * The classifier used for classification in **Support Vector Machine Classifier(SVC)** with *_Linear_* kernel.
   * The data sets before and after **KPCA** is shown below:
   
       ![classification_accuracy_for_before after_kpca linear kernel](https://user-images.githubusercontent.com/36328597/41507805-a9fff0d4-7257-11e8-8c6f-8868337c9890.PNG)
   
   * The classification of the dataset *_before_* **Kernel PCA** is:
   
      | kernel | Accuracy | Execution Time|
      | :---         |     :---:      |          ---: |
      | Linear   | 100     | 0.00200009346    |
      | Radial Basis Function(RBF)     | 100       | 0.0020003318      |
      | Polynomial   | 100     | 0.0010001659   |
      
   * The classification of the dataset *_After_* **Kernel PCA** is:
   
      | kernel | Accuracy | Execution Time|
      | :---         |     :---:      |          ---: |
      | Linear   | 95.55     | 0.0020003318    |
      | Radial Basis Function(RBF)     | 37.77       | 0.00200009346      |
      | Polynomial   | 95.55     | 0.1670093536   |



### Conclusion 

   * By performing **KPCA** with three different kernels (linear,rbf,polynomial) on the iris data set.
   * since, the initial two Principal Components(PC'S) has more variance ratio. we selected two only.
   * Initially the dataset contains the dimensions **150 X 5** is drastically reduced to **150 X 3** dimensions including label.
   * The classification has varied a lot according to the kernel choosen.
   

## License

This project is licensed under the **MIT** License - see the [LICENSE.md](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/LICENSE.md)

