# src/knn_classifier.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA

def plot_confusion_matrix(cm, labels, outpath):
    """
    Plot and save a confusion matrix image.
    cm : 2D array (n_classes x n_classes)
    labels : list of label names (strings)
    outpath : path to save PNG
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center")

    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_decision_boundary_2d(X2, y, clf, labels, outpath, title='Decision boundary (2D)'):
    """
    Plot a decision boundary for a classifier trained on 2D data.
    X2 : (n_samples, 2) array (2D data for plotting)
    y : (n_samples,) array (class labels as ints)
    clf : classifier with .predict that accepts 2D points
    labels : list/array of label names (strings), length == n_classes
    outpath : path to save PNG
    """
    x_min, x_max = X2[:, 0].min() - 1.0, X2[:, 0].max() + 1.0
    y_min, y_max = X2[:, 1].min() - 1.0, X2[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # predict grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(len(labels)+1)-0.5, cmap=cmap)

    # scatter points colored by class
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap=cmap, s=40, edgecolor='k')

    # build legend with label names and proper colors
    classes = np.unique(y)
    colors = [cmap(i) for i in classes]
    legend_patches = [Patch(facecolor=colors[i], edgecolor='k', label=labels[i]) for i in classes]
    plt.legend(handles=legend_patches, title='Classes')

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    os.makedirs('outputs', exist_ok=True)

    # load dataset
    data = load_iris()
    X = data.data
    y = data.target
    target_names = data.target_names.tolist()

    # standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # grid search over k (1..20)
    param_grid = {'n_neighbors': list(range(1, 21))}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_k = grid.best_params_['n_neighbors']
    print(f"Best k (CV): {best_k}, CV score: {grid.best_score_:.4f}")

    clf = grid.best_estimator_
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(cm, target_names, 'outputs/confusion_matrix.png')

    # Decision boundary: reduce to 2D with PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    # train a KNN on the 2D projection for visualization
    clf2 = KNeighborsClassifier(n_neighbors=best_k)
    # Use the full dataset projection and labels so the visualization shows overall boundaries
    clf2.fit(X2, y)
    plot_decision_boundary_2d(
        X2, y, clf2, labels=target_names,
        outpath='outputs/decision_boundary_pca.png',
        title=f'KNN decision boundary (k={best_k})'
    )

    # Save best k and accuracy to a small text file
    with open('outputs/summary.txt', 'w') as f:
        f.write(f"best_k={best_k}\n")
        f.write(f"cv_score={grid.best_score_:.4f}\n")
        f.write(f"test_accuracy={acc:.4f}\n")

    print("Outputs saved to the 'outputs' directory:")
    print(" - outputs/confusion_matrix.png")
    print(" - outputs/decision_boundary_pca.png")
    print(" - outputs/summary.txt")

if __name__ == '__main__':
    main()
