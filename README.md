# Thai Tran EE-399-HW-3 
## Abstract
This project involves an analysis of the MNIST dataset using Singular Value Decomposition (SVD) to gain insights into the structure of digit images. The singular value spectrum is examined to determine the necessary number of modes for good image reconstruction, and the interpretation of U, Σ, and V matrices in SVD is discussed. Linear classifiers, specifically Linear Discriminant Analysis (LDA), are built to identify and classify digits in the training set. The performance of the classifiers is evaluated for different digit pairs, and the accuracy of separation using LDA, SVM, and decision tree classifiers is quantified. Findings from this project provide insights into the effectiveness of different classifiers for identifying digits in the MNIST dataset, and shed light on the separability of different digit pairs using various classification techniques.

## Section I Introduction and Overview
The MNIST dataset, a widely used benchmark in the field of machine learning, consists of a large collection of grayscale images of handwritten digits ranging from 0 to 9. This dataset has been extensively studied to develop and evaluate various classification algorithms. In this project, we perform an analysis of the MNIST dataset using Singular Value Decomposition (SVD), a powerful mathematical technique that can reveal the underlying structure and characteristics of the data. By reshaping the digit images into column vectors and constructing a data matrix, we can apply SVD to extract valuable information and insights from the dataset. The main objectives of this analysis are to understand the singular value spectrum of the digit images, determine the necessary number of modes for good image reconstruction (i.e., the rank of the digit space), and interpret the U, Σ, and V matrices obtained from SVD. Additionally, we will visualize the data projected onto selected V-modes using a 3D plot, and build linear classifiers, specifically LDA, to identify and classify digits in the training set. Furthermore, we will investigate the separability of different digit pairs using LDA, SVM, and decision tree classifiers, and compare their performance in terms of accuracy. We will also quantify the accuracy of separation for the hardest and easiest digit pairs, and compare the performance of these classifiers on these pairs. The findings from this analysis will provide insights into the effectiveness of different classifiers for identifying and classifying digits in the MNIST dataset, and shed light on the separability of different digit pairs using various classification techniques.

## Section II Theoretical Background

### Singular Value Decomposition (SVD)
Mathematical technique used to decompose a matrix into a product of three matrices: U, Σ, and V^T (transpose of V). Given an m x n matrix A, SVD can be represented as A = UΣV^T, where U is an m x m orthogonal matrix, Σ is an m x n diagonal matrix containing the singular values of A, and V^T is an n x n orthogonal matrix.
### Principal Component Analysis (PCA)
Dimensionality reduction technique commonly used in data analysis and machine learning. It is a statistical method that transforms a high-dimensional dataset into a lower-dimensional representation while retaining as much of the original data's variability as possible.

### Linear Discriminant Analysis (LDA)
Classification algorithm that aims to find a linear combination of features that maximally separates different classes. LDA is often used for dimensionality reduction and feature extraction in machine learning, as it can project high-dimensional data onto a lower-dimensional space while preserving the class separability. In the context of image classification, LDA can be used to identify and classify digits in the MNIST dataset based on their features.

### Support Vector Machines (SVM) and decision tree classifiers
Classification algorithms that have been used extensively for image classification. SVM aims to find the optimal hyperplane that maximally separates different classes, while decision tree classifiers recursively split the data based on the most informative features until the classes are well-separated.

## Sec. III. Algorithm Implementation and Development

### Singular Value Decomposition (SVD)
used to perform dimensionality reduction on the MNIST dataset, which consists of images of handwritten digits.
```
# Load the MNIST data
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist.data, mnist.target
X = X / 255.0  # Scale the pixel values to [0, 1]
# Reshape each image into a column vector
X_col = X.T
U, Sigma, Vt = np.linalg.svd(X_col, full_matrices=False)
```
### Principal Component Analysis (PCA)
used in the implementation as a technique for dimensionality reduction. It was applied on the MNIST dataset to reduce the high-dimensional image data to a lower-dimensional representation while preserving the most important information in the data.

```
# Perform PCA
pca = PCA(n_components=784)
X_pca = pca.fit_transform(X)
```
### Linear Discriminant Analysis (LDA)
used as a technique for dimensionality reduction and feature extraction, followed by classification. It was applied on the reduced-dimensional data obtained from PCA to further reduce the feature space and enhance the discriminative power of the data for classification.

```
# Train LDA classifier
lda = LDA()
lda.fit(X_train, y_train)
# Predict on the testing set
y_pred = lda.predict(X_test)
```

### Support Vector Machines (SVM) and decision tree classifiers
used as the classification algorithms on the transformed data obtained from PCA and LDA
```
# Initialize SVM classifier
svm = SVC()
# Initialize Decision Tree classifier
dtc = DecisionTreeClassifier()
# Train SVM classifier
svm.fit(X_train, y_train)
# Train Decision Tree classifier
dtc.fit(X_train, y_train)
# Predict on test data using SVM
y_pred_svm = svm.predict(X_test)
# Predict on test data using Decision Tree
y_pred_dtc = dtc.predict(X_test)
# Calculate accuracy for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
```

## Sec. IV. Computational Results
### Problem set 1 
#### number 1
Loads the MNIST dataset using fetch_openml function and scales the pixel values of the images to [0, 1]. The images are reshaped into column vectors and then subjected to Singular Value Decomposition (SVD) using np.linalg.svd function. The resulting U, Sigma, and Vt matrices represent the left singular vectors, singular values, and right singular vectors, respectively. The full_matrices parameter is set to False, indicating reduced dimensions. The shape of the U matrix is printed using a print statement.
```
U shape: (784, 784)
```

#### numbner 2
![image](https://user-images.githubusercontent.com/129792715/234149954-2dc2dd6c-b428-4a61-8b4a-c20dc3d7e8ef.png)
![image](https://user-images.githubusercontent.com/129792715/234149970-bd1b708c-2789-49c9-915d-44a3cd39ea5e.png)

The first plot shows the singular values of the dataset, with the x-axis representing the index of the singular value and the y-axis representing the value of the singular value. The second plot shows the cumulative sum of the squared singular values, normalized by the sum of squared singular values, with the x-axis representing the number of modes (singular values) and the y-axis representing the cumulative energy.

Based on the cumulative energy plot, the rank (r) of the digit space is determined as the index at which the cumulative energy exceeds or equals 0.9. This value of r is printed using the print statement. The rank indicates the number of singular values needed to capture 90% of the energy in the dataset.

```
The rank r of the digit space is 53
```
