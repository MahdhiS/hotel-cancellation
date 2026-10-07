import pandas as pd

_SEASON = {
    "December": "Winter",
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
}


def add_features(df):
    out = df.copy()
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["total_guests"] = out["adults"] + out["children"] + out["babies"]
    out["has_kids"] = ((out["children"] + out["babies"]) > 0).astype(int)
    out["season"] = out["arrival_date_month"].map(_SEASON)
    out["arrival_day_of_week"] = pd.to_datetime(out["arrival_date"]).dt.day_name()
    return out
