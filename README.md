# Kernel PCA Using Different Kernels With Classification

# Dimensionality reduction and classification on [Hyperspectral Image](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) Using Python
   
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

      ![indian_pines_varianve_ratio](https://user-images.githubusercontent.com/36328597/41495831-56fff622-714e-11e8-87ab-731c11d14bab.JPG)
      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * It second result is a scatter plot for the **Pricipal Components** is :

      ![indian_pines_after_pca_with_2pc](https://user-images.githubusercontent.com/36328597/41495958-603d0baa-7151-11e8-9c7c-c7452b2fb6a8.JPG)


   * The dimensionally reduced file is saved to [csvfile](
https://github.com/syamkakarla98/Dimensionality-reduction-and-classification-on-Hyperspectral-Images-Using-Python/blob/master/indian_pines_after_pca.csv).
 
1. The output of kernel PCA with **_Radial Basis Function(RBF)_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Radial Basis Function(RBF)_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      ![indian_pines_varianve_ratio](https://user-images.githubusercontent.com/36328597/41495831-56fff622-714e-11e8-87ab-731c11d14bab.JPG)
      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * It second result is a scatter plot for the **Pricipal Components** is :

      ![indian_pines_after_pca_with_2pc](https://user-images.githubusercontent.com/36328597/41495958-603d0baa-7151-11e8-9c7c-c7452b2fb6a8.JPG)


   * The dimensionally reduced file is saved to [csvfile](
https://github.com/syamkakarla98/Dimensionality-reduction-and-classification-on-Hyperspectral-Images-Using-Python/blob/master/indian_pines_after_pca.csv).    
   
1. The output of kernel PCA with **_Polynomial_** kernel : 
     * The Explained variance Ratio of the principal components using kernel PCA with **_Polynomial_** kernel and result is shown in bargraph for **4 Pricipal Components** according to their _variance ratio's_ :

      ![indian_pines_varianve_ratio](https://user-images.githubusercontent.com/36328597/41495831-56fff622-714e-11e8-87ab-731c11d14bab.JPG)
      
    Since, The initial two principal components have high variance. So, we selected the first two principal components.
      
      * It second result is a scatter plot for the **Pricipal Components** is :

      ![indian_pines_after_pca_with_2pc](https://user-images.githubusercontent.com/36328597/41495958-603d0baa-7151-11e8-9c7c-c7452b2fb6a8.JPG)


   * The dimensionally reduced file is saved to [csvfile](
https://github.com/syamkakarla98/Dimensionality-reduction-and-classification-on-Hyperspectral-Images-Using-Python/blob/master/indian_pines_after_pca.csv).

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

