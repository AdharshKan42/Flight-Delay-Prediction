from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from data_processing import process_data
from time import perf_counter


def classify_points(X_train, X_test, y_train, y_test):
    model = svm.LinearSVC(max_iter=10000)

    training_start_time = perf_counter()

    model.fit(X_train, y_train)

    training_end_time = perf_counter()

    print(f"Model took {training_end_time - training_start_time:.2f} seconds to train.")

    print(f"Test Classification Error: {(1 - model.score(X_test, y_test))*100:.2f}")
    print(f"Train Classification Accuracy: {(model.score(X_train, y_train))*100:.2f}")
    print(f"Test Classification Accuracy: {(model.score(X_test, y_test))*100:.2f}")


X_train, X_test, y_train, y_test, categories_mapping = process_data(
    "star_classification.csv"
)

print("Running SVM")

classify_points(X_train, X_test, y_train, y_test)
