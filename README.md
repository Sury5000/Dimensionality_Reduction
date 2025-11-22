# Dimensionality Reduction – PCA on MNIST Dataset

This project focuses on applying **Principal Component Analysis (PCA)** to the **MNIST handwritten digits dataset**.  
The goal of this analysis is to understand how high-dimensional image data (784 features) can be reduced to fewer dimensions while still preserving most of the variance in the dataset.

---

## 1. Dataset Description

- **Dataset:** MNIST handwritten digits  
- **Source:** `fetch_openml('mnist_784')`
- **Number of samples:** 70,000  
- **Image size:** 28 × 28  
- **Number of features:** 784 (flattened pixel values)  
- The target variable `y` contains digit labels (0–9).  
- The feature matrix `X` contains pixel intensity values.

Code used:

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
```

---

## 2. Applying PCA (2 Components)

PCA was first applied by reducing the original 784-dimensional features down to **2 principal components**.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

This transforms each digit image into a point in a 2D space.

---

## 3. PCA Components

The principal directions were obtained using:

```python
pca.components_
```

Each component is a vector of length 784 representing the weights assigned to each pixel.

---

## 4. Explained Variance Ratio

The variance retained by each of the two principal components was displayed:

```python
print(pca.explained_variance_ratio_)
```

This shows how much of the original dataset’s information is captured by the reduced representation.

---

## 5. Determining Components Required for 95% Variance

A full PCA (no fixed number of components) was fitted to compute cumulative variance:

```python
import numpy as np
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
```

Steps performed:

- Computed the cumulative sum of explained variance  
- Identified the minimum number of PCA components needed to preserve **95% of total variance**  
- Stored that number in variable `d`

---

## 6. Summary of Work

- Loaded MNIST dataset  
- Reduced dimensionality from 784 to 2 components  
- Viewed PCA directions  
- Checked explained variance ratios  
- Calculated number of components needed for 95% information retention  

This notebook demonstrates the use of PCA for understanding variance and reducing dimensionality in high-dimensional datasets like MNIST.

