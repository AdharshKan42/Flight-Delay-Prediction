from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import process_data

from time import perf_counter

print("Preparing data...")
X_train, X_test, y_train, y_test, categories_mapping = process_data(
    "star_classification.csv"
)

print("Training model...")
model = LogisticRegression()

training_start_time = perf_counter()

model.fit(X_train, y_train)

training_end_time = perf_counter()

print(f"Model took {training_end_time - training_start_time} seconds to train.")

y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100}%")

sns.pairplot(X_test)

plt.show()
