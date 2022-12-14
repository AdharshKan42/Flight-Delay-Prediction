import pandas as pd

months = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

years = [2018, 2019]

data = pd.DataFrame()

for year in years:
    for month in months:
        df = pd.read_csv(f"data/{month}_{year}.csv")
        data = pd.concat([data, df])

columns = [
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Reporting_Airline",
    "OriginAirportID",
    "Origin",
    "DestAirportID",
    "Dest",
    "DepDelay",
    "Cancelled",
]

data.to_csv(
    "flight_data.csv",
    index=False,
    columns=columns,
)
