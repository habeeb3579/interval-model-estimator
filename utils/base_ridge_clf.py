from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def tune_ridge_classifier(X_train, y_train, X_test, y_test, param_grid, plot=True, label=False):
    """
    Tune a Ridge Classifier model using GridSearchCV, then test on a test set.

    Parameters:
    - X_train: array-like, shape (n_samples, n_features)
        Training data.
    - y_train: array-like, shape (n_samples,)
        Target labels for training data.
    - X_test: array-like, shape (n_samples, n_features)
        Test data.
    - y_test: array-like, shape (n_samples,)
        Target labels for test data.

    Returns:
    - accuracy: float
        Accuracy of the tuned model on the test set.
    - cm: array-like, shape (n_classes, n_classes)
        Confusion matrix of the tuned model on the test set.
    """

    ridge_classifier = RidgeClassifier()
    #param_grid = {'alpha': np.logspace(-5, 5, 20)}

    grid_search = GridSearchCV(ridge_classifier, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']

    ridge_classifier_best = RidgeClassifier(alpha=best_alpha)
    ridge_classifier_best.fit(X_train, y_train)

    y_pred = ridge_classifier_best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)


    if plot:
        if label:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels(y_train))
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    return accuracy, cm

