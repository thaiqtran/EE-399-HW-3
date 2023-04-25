# Thai Tran EE-399-HW-3 
## Abstract
This project involves an analysis of the MNIST dataset using Singular Value Decomposition (SVD) to gain insights into the structure of digit images. The singular value spectrum is examined to determine the necessary number of modes for good image reconstruction, and the interpretation of U, Σ, and V matrices in SVD is discussed. Linear classifiers, specifically Linear Discriminant Analysis (LDA), are built to identify and classify digits in the training set. The performance of the classifiers is evaluated for different digit pairs, and the accuracy of separation using LDA, SVM, and decision tree classifiers is quantified. Findings from this project provide insights into the effectiveness of different classifiers for identifying digits in the MNIST dataset, and shed light on the separability of different digit pairs using various classification techniques.

## Section I Introduction and Overview
The MNIST dataset, a widely used benchmark in the field of machine learning, consists of a large collection of grayscale images of handwritten digits ranging from 0 to 9. This dataset has been extensively studied to develop and evaluate various classification algorithms. In this project, we perform an analysis of the MNIST dataset using Singular Value Decomposition (SVD), a powerful mathematical technique that can reveal the underlying structure and characteristics of the data. By reshaping the digit images into column vectors and constructing a data matrix, we can apply SVD to extract valuable information and insights from the dataset. The main objectives of this analysis are to understand the singular value spectrum of the digit images, determine the necessary number of modes for good image reconstruction (i.e., the rank of the digit space), and interpret the U, Σ, and V matrices obtained from SVD. Additionally, we will visualize the data projected onto selected V-modes using a 3D plot, and build linear classifiers, specifically LDA, to identify and classify digits in the training set. Furthermore, we will investigate the separability of different digit pairs using LDA, SVM, and decision tree classifiers, and compare their performance in terms of accuracy. We will also quantify the accuracy of separation for the hardest and easiest digit pairs, and compare the performance of these classifiers on these pairs. The findings from this analysis will provide insights into the effectiveness of different classifiers for identifying and classifying digits in the MNIST dataset, and shed light on the separability of different digit pairs using various classification techniques.

## Section II Theoretical Background

### Singular Value Decomposition (SVD)
Mathematical technique used to decompose a matrix into a product of three matrices: U, Σ, and V^T (transpose of V). Given an m x n matrix A, SVD can be represented as A = UΣV^T, where U is an m x m orthogonal matrix, Σ is an m x n diagonal matrix containing the singular values of A, and V^T is an n x n orthogonal matrix.

### Linear Discriminant Analysis (LDA)
Classification algorithm that aims to find a linear combination of features that maximally separates different classes. LDA is often used for dimensionality reduction and feature extraction in machine learning, as it can project high-dimensional data onto a lower-dimensional space while preserving the class separability. In the context of image classification, LDA can be used to identify and classify digits in the MNIST dataset based on their features.

### Support Vector Machines (SVM) and decision tree classifiers
Classification algorithms that have been used extensively for image classification. SVM aims to find the optimal hyperplane that maximally separates different classes, while decision tree classifiers recursively split the data based on the most informative features until the classes are well-separated.

## Sec. III. Algorithm Implementation and Development

