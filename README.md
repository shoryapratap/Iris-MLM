# 🌸 Iris Dataset Classification using K-Nearest Neighbors (KNN)

This project demonstrates a complete pipeline to classify the famous Iris dataset using the **K-Nearest Neighbors (KNN)** algorithm. The project includes data exploration, preprocessing, training, evaluation, and visualization (including decision boundaries).

---

## 📁 Project Structure


---

## 📚 Dataset

- **Source**: [Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- **Description**: The dataset contains 150 samples of iris flowers from three species (Setosa, Versicolor, Virginica), each described by four features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width

---

## 🚀 Workflow Overview

### 1. **Import Required Libraries**

Uses standard libraries like:
- `numpy`, `matplotlib`
- `scikit-learn` modules

### 2. **Load and Explore Data**

- Load the dataset using `sklearn.datasets`.
- Display feature and target names.
- Show sample data.

### 3. **Preprocess the Data**

- Split the data into training and testing sets.
- Standardize features using `StandardScaler`.

### 4. **Train the KNN Model**

- Use `KNeighborsClassifier` with `k=3`.
- Fit the model on the training data.

### 5. **Make Predictions**

- Predict classes on the test set.

### 6. **Evaluate the Model**

- Generate a **confusion matrix**.
- Print **classification report** (precision, recall, f1-score).
- Show **accuracy** score.

### 7. **Visualize Confusion Matrix**

- Display confusion matrix using `ConfusionMatrixDisplay`.

### 8. **Visualize Decision Boundaries**

- Train another KNN model on **only 2 features** (sepal length and width).
- Visualize decision regions using a **mesh grid** and `matplotlib`.

---

## 📊 Results

- **Accuracy**: ~97% on the test set (may vary slightly due to train-test split).
- Clear **decision boundaries** for visual intuition of KNN classification.

---

## 📦 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy matplotlib scikit-learn


📌 How to Run

python iris_knn_classifier.py
