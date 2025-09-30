# k-Nearest-Neighbors-KNN-Classification-
KNN Classifier on Iris Dataset
📌 Project Overview
This project implements a K-Nearest Neighbors (KNN) classifier on the classic Iris dataset using scikit-learn.
It includes data preprocessing, model selection via cross-validation, evaluation, and visualization of results.
The goal is to:
Automatically find the best value of k using cross-validation.Train and evaluate the model on a hold-out test set.
Report accuracy, confusion matrix, and classification metrics.Optionally visualize decision boundaries and confusion matrix.
⚙️ Features
✅ Data Handling
Loads the built-in Iris dataset from scikit-learn.
Standardizes features using StandardScaler.
Stratified train/test split for balanced class distribution.
✅ Model Training
Uses GridSearchCV to search for the best k in the range 1 to 20.
Selects the model with the highest cross-validated accuracy.
✅ Evaluation
Prints best k value and CV accuracy.
Evaluates on test set: accuracy, confusion matrix, classification report.
Saves a summary file (outputs/summary.txt) with metrics.
✅ Visualization (extra, optional)
Confusion Matrix Plot (saved as PNG).
2D Decision Boundary Plot using PCA for dimensionality reduction.
✅ Output Management
All outputs (plots + summary) saved in an outputs/ folder (created automatically).
📂 Project Structure
knn-iris-classifier/
│
├── src/
│   └── knn_classifier.py   # main script
├── outputs/                # generated results (ignored in Git)
│   ├── confusion_matrix.png
│   ├── decision_boundary_pca.png
│   └── summary.txt
├── requirements.txt
└── README.md

🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/knn-iris-classifier.git
cd knn-iris-classifier


Install dependencies:

pip install -r requirements.txt


Run the script:

python src/knn_classifier.py


Check results in the outputs/ folder.

📊 Example Output (console)
Best k (CV): 13, CV score: 0.9667
Test accuracy: 1.0000
Classification report:
               precision    recall  f1-score   support
    setosa       1.00      1.00      1.00        10
versicolor       1.00      1.00      1.00        10
 virginica       1.00      1.00      1.00        10
Confusion matrix:
 [[10  0  0]
  [ 0 10  0]
  [ 0  0 10]]

📦 Requirements

Python 3.8+

numpy

matplotlib

scikit-learn

You can install them via:

pip install -r requirements.txt

✨ Key Learnings

How to implement and tune a KNN classifier.

Importance of standardizing features.

Using cross-validation for hyperparameter tuning.

Visualizing decision boundaries in reduced dimensions.

Evaluating models using confusion matrix and classification report.
