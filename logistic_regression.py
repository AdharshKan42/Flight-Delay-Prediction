from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import process_data
import pandas as pd


from time import perf_counter

print("Preparing data...")
X_train, X_test, y_train, y_test, categories_mapping = process_data(
    "star_classification.csv"
)

print("Training model...")
model = LogisticRegression(max_iter=1500)

training_start_time = perf_counter()

model.fit(X_train, y_train)

training_end_time = perf_counter()

print(f"Model took {training_end_time - training_start_time:.2f} seconds to train.")

y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100}%")

X_test['Class'] = y_test

sns.pairplot(X_test, hue="Class")

plt.show()
