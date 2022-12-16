from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from data_processing import process_data


def classify_points(X_train, X_test, y_train, y_test):
    model = svm.LinearSVC()

    model.fit(X_train, y_train)

    print(f"Test Classification Error: {1 - model.score(X_test, y_test):.2f}")
    print(f"Train Classification Accuracy: {model.score(X_train, y_train):.2f}")
    print(f"Test Classification Accuracy: {model.score(X_test, y_test):.2f}")

    predictions = model.predict(X_test)

    for i in range(len(predictions)):
        if predictions[i] == 0 or predictions[i] == -1:
            # ax.plot(X_test[i][0], X_test[i][1], c="red", marker="x")
            pass
        elif predictions[i] == 1:
            # ax.plot(X_test[i][0], X_test[i][1], c="blue", marker="x")
            pass


# load dataset here
X_train, X_test, y_train, y_test, categories_mapping = process_data(
    "star_classification.csv"
)

print("Running SVM on Dataset 1")

# run program
classify_points(X_train, X_test, y_train, y_test)
