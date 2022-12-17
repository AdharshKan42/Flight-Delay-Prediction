import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(filepath: str):
    print("Reading data...")
    data = pd.read_csv(filepath)

    print(f"Number of Rows in Original Data: {data.shape[0]}")

    print("Removing rows with no class identified...")

    # Removes any rows with no value in the table (N/A)
    processed_data = data.dropna(
        axis=0,
        how="any",
        subset=[
            "class",
        ],
    )

    print(f"Number of Rows in Processed Data: {processed_data.shape[0]}")
    print(f"Number of Rows Removed: {data.shape[0] - processed_data.shape[0]}")

    categories_mapping = {"GALAXY": 0, "QSO": 1, "STAR": 2}

    print("Encoding categories to index values...")

    processed_data["class"].replace(
        categories_mapping.keys(),
        categories_mapping.values(),
        inplace=True,
    )

    features = processed_data[["u", "g", "r", "i", "z", "redshift"]]

    categories = processed_data["class"]
    category_counts = []

    print("Number of Instances in each category:")
    for (k, v) in categories_mapping.items():
        category_count = categories.to_list().count(v)
        category_counts.append(category_count)
        print(f"{k}: {category_count}")

    print("Splitting train and test data...")

    train_features, test_features, train_categories, test_categories = train_test_split(
        features, categories, test_size=0.25, random_state=10, shuffle=True
    )

    train_categories = train_categories.values
    test_categories = test_categories.values

    print("Done processing data!")

    return (
        train_features,
        test_features,
        train_categories,
        test_categories,
        categories_mapping,
    )
