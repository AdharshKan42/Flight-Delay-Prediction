from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import process_data
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd


print("Preparing data")
X_train, X_test, y_train, y_test, classes_mapping = process_data(
    "star_classification.csv"
)

print("Training model")
model = LogisticRegression()
log_model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print(f"Accuracy: {test_accuracy * 100}%")

X_test['Class'] = y_test
# y_test_col = y_test.reshape(-1,1)
# df_y_test = pd.DataFrame(y_test_col, columns = ['class'])
# print(df_y_test.shape)
# print(X_test.shape)
# # print(df_y_test.head)
print(X_test.head)

# result = pd.concat([X_test, df_y_test], axis = 1)
# print(result.head)

sns.pairplot(X_test, hue = "Class")

# print(X_test.u)
# print(y_test)
# sns.regplot(X_test, y_test, logistic=True)

# X_test_ug = X_test[["u", "g"]].to_numpy()

# print(X_test_ug)

# print(X_test.head)
# display = DecisionBoundaryDisplay.from_estimator(
#     log_model,
#     X_test,
#     response_method="predict",
#     label_x="X_1",
#     label_y="X_2",
#     alpha=0.3,
# )

# display.ax_.scatter(X_test["u"], X_test["g"], c=X_test["u"], edgecolor="k")


# X_test_ug = X_test[["u", "g"]]

# _, ax = plt.subplots(figsize=(6, 3))
# DecisionBoundaryDisplay.from_estimator(
#     log_model,
#     X_test,
#     cmap=plt.cm.Paired,
#     ax=ax,
#     response_method="predict",
#     plot_method="pcolormesh",
#     shading="auto",
#     xlabel="Sepal length",
#     ylabel="Sepal width",
#     eps=0.5,
# )

# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k", cmap=plt.cm.Paired)


plt.show()
