import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(filepath: str):
    data = pd.read_csv(filepath)

    print(f"Number of Rows in Original Data: {data.shape[0]}")

    # Removes any rows with no value in the table for
    # time delayed for departure (DepDelay)
    processed_data = data.dropna(
        axis=0,
        how="any",
        subset=[
            "DepDelay",
        ],
    )

    print(f"Number of Rows in Processed Data: {processed_data.shape[0]}")
    print(f"Number of Rows Removed: {data.shape[0] - processed_data.shape[0]}")

    airlines = processed_data["Reporting_Airline"].unique()
    airlines_mapping = dict([(v, k) for k, v in dict(enumerate(airlines)).items()])

    processed_data["Reporting_Airline"].replace(
        airlines_mapping.keys(),
        airlines_mapping.values(),
        inplace=True,
    )

    origin_airports = processed_data["Origin"].unique()
    origin_ids = processed_data["OriginAirportID"].unique()
    airports_mapping = dict(zip(origin_airports, origin_ids))

    dest_airports = processed_data["Dest"].unique()
    dest_ids = processed_data["DestAirportID"].unique()
    airports_mapping.update(dict(zip(dest_airports, dest_ids)))

    features = processed_data[
        [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "Reporting_Airline",
            "OriginAirportID",
            "DestAirportID",
        ]
    ]

    actual_delay = processed_data["DepDelay"]

    train_features, test_features, train_delay, test_delay = train_test_split(
        features, actual_delay, test_size=0.25, random_state=10, shuffle=True
    )

    train_delay = train_delay.values.reshape(-1, 1)
    test_delay = test_delay.values.reshape(-1, 1)

    return (
        train_features,
        test_features,
        train_delay,
        test_delay,
        airlines_mapping,
        airports_mapping,
    )
