from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from data_processing import process_data

print("Preparing data")
X_train, X_test, y_train, y_test, airlines_mapping, airplanes_mapping = process_data(
    "flight_data.csv"
)

print("Training model")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print(f"Accuracy: {test_accuracy * 100}%")

fig, axs = plt.subplots(2, 3)

for i in range(X_test.shape[1]):
    ax = axs[(i // 3)][(i % 3)]
    ax.scatter(X_test.iloc[:, i].values[:5], y_pred[:5], color="blue")
    ax.scatter(X_test.iloc[:, i].values[:5], y_test[:5], color="red")

plt.show()
