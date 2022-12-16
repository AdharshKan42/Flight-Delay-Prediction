from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import process_data

print("Preparing data")
X_train, X_test, y_train, y_test, classes_mapping = process_data(
    "star_classification.csv"
)

print("Training model")
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print(f"Accuracy: {test_accuracy * 100}%")

sns.pairplot(X_test)

plt.show()
