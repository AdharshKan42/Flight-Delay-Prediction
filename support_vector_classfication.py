from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from data_processing import process_data


def prep_y_data(arr):
    return np.array([i[0] for i in arr])


def classify_points(X_train, X_test, Y_train, Y_test, kernel, ax):

    model = svm.SVC(kernel=kernel)

    model.fit(X_train, Y_train)

    print(f"{kernel.capitalize()} Kernel:")
    ax.set_title(f"{kernel.capitalize()} Kernel:")

    print(f"Test Classification Error: {1 - model.score(X_test, Y_test):.2f}")
    print(f"Train Classification Accuracy: {model.score(X_train, Y_train):.2f}")
    print(f"Test Classification Accuracy: {model.score(X_test, Y_test):.2f}")

    ax.scatter(X_train[:, 0], X_train[:, 1], c="green", edgecolor="k", marker="o")

    predictions = model.predict(X_test)

    for i in range(len(predictions)):
        if predictions[i] == 0 or predictions[i] == -1:
            ax.plot(X_test[i][0], X_test[i][1], c="red", marker="x")
        elif predictions[i] == 1:
            ax.plot(X_test[i][0], X_test[i][1], c="blue", marker="x")


fig, axs = plt.subplots(2, 3)

# load dataset here
X_train, X_test, y_train, y_test, categories_mapping = process_data(
    "star_classification.csv"
)

print("Running SVM on Dataset 1")

# run program
classify_points(X_train, X_test, y_train, y_test, "linear", axs[0, 0])
classify_points(X_train, X_test, y_train, y_test, "rbf", axs[0, 1])
classify_points(X_train, X_test, y_train, y_test, "poly", axs[0, 2])
