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

#### number 2
![image](https://user-images.githubusercontent.com/129792715/234149954-2dc2dd6c-b428-4a61-8b4a-c20dc3d7e8ef.png)
![image](https://user-images.githubusercontent.com/129792715/234149970-bd1b708c-2789-49c9-915d-44a3cd39ea5e.png)

The first plot shows the singular values of the dataset, with the x-axis representing the index of the singular value and the y-axis representing the value of the singular value. The second plot shows the cumulative sum of the squared singular values, normalized by the sum of squared singular values, with the x-axis representing the number of modes (singular values) and the y-axis representing the cumulative energy.

Based on the cumulative energy plot, the rank (r) of the digit space is determined as the index at which the cumulative energy exceeds or equals 0.9. This value of r is printed using the print statement. The rank indicates the number of singular values needed to capture 90% of the energy in the dataset.

```
The rank r of the digit space is 53
```
#### number 3) 
What is the interpretation of the U, Σ, and V matrices?

Answer: U (Left singular vectors): U is an orthogonal matrix whose columns are the left singular vectors of the data matrix. In this code, U is obtained using svd.fit_transform(X_col). Each column of U represents a reduced-dimensional representation of the original data, where the rows of U are the features (or latent variables) extracted from the data. These left singular vectors represent the directions in the original feature space along which the data varies the most.

Σ (Singular values): Σ is a diagonal matrix containing the singular values, which represent the magnitude of the variation captured by each singular vector. In this code, Σ is obtained using svd.singular_values_. The singular values are sorted in descending order, with the first singular value corresponding to the direction of greatest variation in the data, the second singular value corresponding to the second greatest variation, and so on. The singular values provide information about the importance of each singular vector in capturing the overall variation in the data.

V (Right singular vectors): V is an orthogonal matrix whose rows are the right singular vectors of the data matrix. In this code, V is obtained using svd.components_. Each row of V represents a reduced-dimensional representation of the original features, where the columns of V are the coefficients that map the reduced-dimensional representations back to the original feature space. The right singular vectors represent the contribution of each feature to the original data

#### number 4)
![image](https://user-images.githubusercontent.com/129792715/234150472-c39debef-87e5-4426-aaec-9004be5faf3e.png)

The code performs Principal Component Analysis (PCA) on the MNIST dataset using the PCA class from sklearn.decomposition module. It then selects three V-modes (columns) from the transformed data, denoted as v_selected. Next, a 3D scatter plot is created using matplotlib's Axes3D module, where the x-axis represents the first selected V-mode, the y-axis represents the second selected V-mode, and the z-axis represents the third selected V-mode. The scatter plot shows the projection of the data points onto the selected V-modes, with different colors representing different digit labels (y). The plot helps visualize the distribution of data points in a reduced-dimensional space.
Answer: U (Left singular vectors): U is an orthogonal matrix whose columns are the left singular vectors of the data matrix. In this code, U is obtained using svd.fit_transform(X_col). Each column of U represents a reduced-dimensional representation of the original data, where the rows of U are the features (or latent variables) extracted from the data. These left singular vectors represent the directions in the original feature space along which the data varies the most.

### Problem set 2
#### number 1)
Using Linear Discriminant Analysis (LDA) on the PCA transformed data for two digits (3 and 8) from the MNIST dataset. It combines the PCA transformed data for the two digits and splits it into training and testing sets. Then, it trains an LDA classifier using the training set and predicts the labels for the testing set. Finally, it calculates and prints the accuracy of the LDA classifier in percentage, which represents the percentage of correct predictions on the testing set
```
Accuracy: 96.13%
```
#### number 2)
Performs digit classification using Linear Discriminant Analysis (LDA) on the PCA transformed data for three digits (3, 8, and 0) from the MNIST dataset. It combines the PCA transformed data for the three digits and splits it into training and testing sets. Then, it trains an LDA classifier using the training set and predicts the labels for the testing set. Finally, it calculates and prints the accuracy of the LDA classifier in percentage, which represents the percentage of correct predictions on the testing set.
```
Accuracy: 95.64%
```
#### number 3&4

```
Most Difficult Pair: Digit 5 and Digit 8: Accuracy = 0.9498
Easiest Pair: Digit 6 and Digit 7: Accuracy = 0.9969
```

#### number 5
The SVM classifier achieved an accuracy of 0.9764 on the test data, while the Decision Tree classifier achieved an accuracy of 0.8708. This suggests that the SVM classifier performed better in terms of accuracy on the given dataset. Further analysis and evaluation, such as cross-validation, hyperparameter tuning, and model comparison, could be performed for a more comprehensive assessment of the classifiers' performance and to select the best model for the task. It's important to note that actual accuracy values may vary depending on the specific dataset and train-test split used, and further evaluation is recommended for robust conclusions.

#### number 6
The performance of three classifiers, Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), and Decision Tree, was evaluated on two pairs of classes - a hard pair and an easy pair - using the accuracy metric. Among these classifiers, SVM consistently outperformed the other classifiers in terms of accuracy for both the hard and easy pairs of classes. These results suggest that SVM is a promising classifier for the given dataset. However, further evaluation and comparison with other performance metrics and techniques would be necessary for a comprehensive assessment of the classifiers' performance.

```
Accuracy of LDA on hard pair: 0.9606
Accuracy of SVM on hard pair: 0.9890
Accuracy of Decision Tree on hard pair: 0.9525
Accuracy of LDA on easy pair: 0.9887
Accuracy of SVM on easy pair: 0.9961
Accuracy of Decision Tree on easy pair: 0.9878
Accuracy of LDA on easy pair: 0.9887
```

## Sec. V Conclusion
In this project, we analyzed the MNIST dataset using Singular Value Decomposition (SVD) to gain insights into the structure of digit images. We examined the singular value spectrum to determine the necessary number of modes for good image reconstruction and interpreted the U, Σ, and V matrices obtained from SVD. We also built linear classifiers, specifically LDA, to identify and classify digits in the training set, and evaluated their performance for different digit pairs. We compared the performance of LDA, SVM, and decision tree classifiers in terms of accuracy and quantified the accuracy of separation for the hardest and easiest digit pairs. The findings from this project provide insights into the effectiveness of different classifiers for identifying digits in the MNIST dataset and shed light on the separability of different digit pairs using various classification techniques. Further research could be conducted to explore other machine learning techniques and algorithms for image classification on the MNIST dataset, and to investigate the performance of classifiers on other datasets with different characteristics.
