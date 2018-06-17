# [Kernel PCA Using](https://scholar.google.co.in/scholar?q=kernel+pca+dimensionality+reduction&hl=en&as_sdt=0&as_vis=1&oi=scholart) Different Kernels With Classification using python 

   
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



   * The dimensionally reduced file is saved to [csvfile](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/iris_after_KPCA_using_linear.csv).
 
2. The output of kernel PCA with **_Radial Basis Function(RBF)_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Radial Basis Function(RBF)_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      ![vr_using_rbf_kernel](https://user-images.githubusercontent.com/36328597/41507821-f39ee5c4-7257-11e8-99fc-b90548ff26c3.png)

      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * THe scatter plot for the **2 Pricipal Components** is :

      
      ![kpca_using_rbf_kernel](https://user-images.githubusercontent.com/36328597/41507820-f36374ee-7257-11e8-8459-09ed3b43d1ae.JPG)


      * The dimensionally reduced file is saved to [csvfile](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/iris_after_KPCA_using_poly.csv).    
   
3. The output of kernel PCA with **_Polynomial_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Polynomial_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      ![vr_using_poly_kernel](https://user-images.githubusercontent.com/36328597/41507799-9d72765c-7257-11e8-8564-20e07c526b3f.png)

      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * The scatter plot for the **2 Pricipal Components** is :

      ![kpca_using_poly_kernel](https://user-images.githubusercontent.com/36328597/41507795-9c73fdac-7257-11e8-9e61-8deddc25203d.JPG)



   * The dimensionally reduced file is saved to [csvfile](https://github.com/syamkakarla98/Kernel-PCA-Using-Different-Kernels-With-Classification/blob/master/iris_after_KPCA_using_rbf.csv).

### Conclusion :

   * By performing **PCA** on the corrected indian pines dataset results **100 Principal Components(PC'S)**.
   * since, the initial two Principal Components(PC'S) has **92.01839071674918** variance ratio. we selected two only.
   * Initially the dataset contains the dimensions **21025 X 200** is drastically reduced to **21025 X 2** dimensions.
   * The time taken for classification before and after Principal Component Analysis(PCA) is:
         
         |   Dataset     |   Accuracy    | Time Taken |
         | ------------- |:-------------:| ----------:|
         |  Before PCA   |   72.748890   |  17.6010   |
         |  After PCA    |   60.098187   | 0.17700982 |
       
   * Hence, the **time** has been reduced with a lot of difference and the **classification accuracy(C.A)** also reduced but the  C.A can increased little bit by varying the 'k' value. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/syamkakarla98/Dimensionality-reduction-and-classification-on-Hyperspectral-Images-Using-Python/blob/master/LICENSE.md) file for details

